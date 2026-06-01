#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RNP (Reconstructive Neuron Pruning) baseline defense.
Li et al., "Reconstructive Neuron Pruning for Backdoor Defense",
ICML 2023.

Algorithm:
  1. Unlearn: Maximize model error on a small clean subset (negative CE)
     to force backdoor-correlated neurons to become dormant.
     Operates at the neuron level (MLP fc1 outputs in transformer blocks).
  2. Recover: Minimize model error on the same clean data to restore
     clean task performance.
  3. Prune: Compare neuron activations between the original and recovered
     model. Neurons with the largest activation change are likely
     backdoor-related and are pruned.

For ViT adaptation we target MLP fc1 neurons in the last few transformer
blocks, using get_transformer_blocks() for Swin compatibility.

Usage:
  python baseline_rnp.py --victim_ckpt victim_badnets.pth \
    --attack badnets --trigger_size 4 \
    --data ./data --offline --no_download \
    --unlearn_epochs 5 --recover_epochs 5 --prune_ratio 0.3
"""
import argparse, os, sys, copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Subset

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from deit_tiny_fs_plus_offline import (
    build_vit_model, build_attack, build_transforms, build_dataset,
    NUM_CLASSES, evaluate_clean_acc, evaluate_asr,
    _assert_cifar10_present, _set_offline_env, set_seed, device,
    get_transformer_blocks,
)


# ---------------------------------------------------------------------------
# Activation collection
# ---------------------------------------------------------------------------
def _get_fc1_layers(model, num_blocks):
    """Return the last `num_blocks` (block_index, fc1_linear) pairs."""
    blocks = get_transformer_blocks(model)
    targets = blocks[-num_blocks:] if num_blocks < len(blocks) else blocks
    pairs = []
    for i, blk in enumerate(blocks):
        if blk in targets:
            pairs.append((i, blk.mlp.fc1))
    return pairs


def collect_activations(model, loader, dev, fc1_layers):
    """Run one pass over *loader* and collect mean absolute activations
    for every fc1 neuron in the targeted layers.
    Returns dict: block_idx -> Tensor of shape (hidden_dim,)."""
    accum = {idx: torch.zeros(fc1.out_features, device=dev)
             for idx, fc1 in fc1_layers}
    counts = {idx: 0 for idx, _ in fc1_layers}

    hooks, handles = {}, []

    def _make_hook(idx):
        def hook_fn(_module, _inp, out):
            # out: (B, N, D) or (B, D)
            hooks[idx] = out.detach().abs().mean(dim=tuple(range(out.dim() - 1)))
        return hook_fn

    for idx, fc1 in fc1_layers:
        h = fc1.register_forward_hook(_make_hook(idx))
        handles.append(h)

    model.eval()
    with torch.no_grad():
        for x, y in loader:
            x = x.to(dev)
            out = model(x)
            if isinstance(out, tuple):
                out = (out[0] + out[1]) / 2.0
            for idx in accum:
                accum[idx] += hooks[idx] * x.size(0)
                counts[idx] += x.size(0)

    for h in handles:
        h.remove()

    for idx in accum:
        accum[idx] /= max(counts[idx], 1)
    return accum


# ---------------------------------------------------------------------------
# Unlearn & Recover
# ---------------------------------------------------------------------------
def unlearn(model, loader, dev, epochs, lr, fc1_layers):
    """Maximize CE loss (negative cross-entropy) on clean data to suppress
    backdoor-correlated neurons.  Only fc1 parameters are updated."""
    params = []
    for _, fc1 in fc1_layers:
        params.extend(fc1.parameters())
    # Freeze everything else
    for p in model.parameters():
        p.requires_grad_(False)
    for p in params:
        p.requires_grad_(True)

    opt = optim.SGD(params, lr=lr, momentum=0.9)
    model.train()
    for ep in range(epochs):
        total_loss = 0.0
        n_samples = 0
        for x, y in loader:
            x, y = x.to(dev), y.to(dev)
            out = model(x)
            if isinstance(out, tuple):
                out = (out[0] + out[1]) / 2.0
            loss = -F.cross_entropy(out, y)  # negative → maximize error
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += loss.item() * x.size(0)
            n_samples += x.size(0)
        print(f"  [RNP-unlearn] ep {ep+1}/{epochs} "
              f"neg_loss={total_loss/n_samples:.4f}")


def recover(model, loader, dev, epochs, lr):
    """Fine-tune the full model with standard CE to recover clean accuracy."""
    for p in model.parameters():
        p.requires_grad_(True)
    opt = optim.Adam(model.parameters(), lr=lr)
    model.train()
    for ep in range(epochs):
        total_loss = 0.0
        n_samples = 0
        for x, y in loader:
            x, y = x.to(dev), y.to(dev)
            out = model(x)
            if isinstance(out, tuple):
                out = (out[0] + out[1]) / 2.0
            loss = F.cross_entropy(out, y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += loss.item() * x.size(0)
            n_samples += x.size(0)
        print(f"  [RNP-recover] ep {ep+1}/{epochs} "
              f"loss={total_loss/n_samples:.4f}")


# ---------------------------------------------------------------------------
# Pruning
# ---------------------------------------------------------------------------
def prune_by_activation_change(model, act_orig, act_recovered, prune_ratio):
    """Compute |act_orig - act_recovered| for each neuron.  Prune the top
    `prune_ratio` fraction by zeroing weights in fc1 and fc2."""
    blocks = get_transformer_blocks(model)
    all_diffs = []
    neuron_ids = []  # (block_idx, neuron_idx)

    for idx in sorted(act_orig.keys()):
        diff = (act_orig[idx] - act_recovered[idx]).abs()
        for j in range(diff.numel()):
            all_diffs.append(diff[j].item())
            neuron_ids.append((idx, j))

    if len(all_diffs) == 0:
        return 0

    all_diffs_t = torch.tensor(all_diffs)
    k = max(1, int(len(all_diffs) * prune_ratio))
    threshold = all_diffs_t.topk(k).values[-1].item()

    n_pruned = 0
    prune_sets = {}  # block_idx -> set of neuron indices
    for (bidx, nidx), d in zip(neuron_ids, all_diffs):
        if d >= threshold:
            prune_sets.setdefault(bidx, set()).add(nidx)

    with torch.no_grad():
        for bidx, nidxs in prune_sets.items():
            blk = blocks[bidx]
            fc1 = blk.mlp.fc1
            fc2 = blk.mlp.fc2
            for nidx in nidxs:
                fc1.weight.data[nidx].zero_()
                if fc1.bias is not None:
                    fc1.bias.data[nidx] = 0.0
                fc2.weight.data[:, nidx].zero_()
                n_pruned += 1

    return n_pruned


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser()
    p.add_argument('--victim_ckpt', required=True)
    p.add_argument('--data', default='./data')
    p.add_argument('--dataset', default='cifar10',
                   choices=['cifar10', 'gtsrb', 'tiny_imagenet'])
    p.add_argument('--model', default='deit_tiny',
                   choices=['deit_tiny', 'deit_small', 'deit_base',
                            'vit_b16', 'swin_t'])
    p.add_argument('--offline', action='store_true')
    p.add_argument('--no_download', action='store_true')
    p.add_argument('--clean_fraction', type=float, default=0.10)
    p.add_argument('--batch_size', type=int, default=128)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--target', type=int, default=0)
    p.add_argument('--poison_source_classes', nargs='+', type=int, default=[3])
    # RNP hyperparameters
    p.add_argument('--unlearn_lr', type=float, default=5e-4)
    p.add_argument('--unlearn_epochs', type=int, default=5)
    p.add_argument('--recover_lr', type=float, default=1e-4)
    p.add_argument('--recover_epochs', type=int, default=5)
    p.add_argument('--prune_ratio', type=float, default=0.3)
    p.add_argument('--num_target_blocks', type=int, default=4,
                   help='Number of last transformer blocks to target')
    # Attack args
    p.add_argument('--attack', required=True,
                   choices=['badnets', 'wanet', 'blend', 'trojvit', 'badvit'])
    p.add_argument('--trigger_size', type=int, default=4)
    p.add_argument('--wanet_s', type=float, default=1.0)
    p.add_argument('--wanet_grid', type=int, default=4)
    p.add_argument('--wanet_grid_rescale', type=float, default=1.0)
    p.add_argument('--wanet_cross_ratio', type=float, default=1.0)
    p.add_argument('--blend_alpha', type=float, default=0.2)
    p.add_argument('--blend_pattern', default='noise')
    p.add_argument('--trojvit_center_row', type=int, default=12)
    p.add_argument('--trojvit_center_col', type=int, default=12)
    p.add_argument('--trojvit_radius_patches', type=int, default=1)
    p.add_argument('--trojvit_alpha', type=float, default=0.9)
    p.add_argument('--trojvit_pattern', default='checker')
    p.add_argument('--badvit_mode', default='visible')
    p.add_argument('--badvit_shape', default='cross')
    p.add_argument('--badvit_alpha', type=float, default=0.8)
    p.add_argument('--badvit_patch_size', type=int, default=16)
    p.add_argument('--badvit_thickness', type=int, default=1)
    p.add_argument('--badvit_norm', default='linf')
    p.add_argument('--badvit_eps', type=float, default=8/255)
    args = p.parse_args()

    dataset_name = getattr(args, 'dataset', 'cifar10')
    model_name = getattr(args, 'model', 'deit_tiny')
    num_cls = NUM_CLASSES.get(dataset_name, 10)

    if args.offline:
        _set_offline_env()
        if dataset_name == 'cifar10':
            _assert_cifar10_present(args.data)
    set_seed(args.seed)
    dev = device()

    # ------------------------------------------------------------------
    # Data
    # ------------------------------------------------------------------
    train_tf, test_tf, _, _ = build_transforms(img_size=224,
                                               dataset=dataset_name)
    download = (not args.offline) and (not args.no_download)
    full_train = build_dataset(dataset_name, args.data, transform=train_tf,
                               train=True, download=download)
    test_ds = build_dataset(dataset_name, args.data, transform=test_tf,
                            train=False, download=download)

    import random
    rng = random.Random(args.seed)
    k = int(len(full_train) * args.clean_fraction)
    idxs = list(range(len(full_train)))
    rng.shuffle(idxs)
    clean_subset = Subset(full_train, idxs[:k])
    clean_loader = DataLoader(clean_subset, batch_size=args.batch_size,
                              shuffle=True, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size,
                             shuffle=False, num_workers=4, pin_memory=True)

    # ------------------------------------------------------------------
    # Load victim
    # ------------------------------------------------------------------
    victim = build_vit_model(model_name, pretrained=False,
                             num_classes=num_cls).to(dev)
    ckpt = torch.load(args.victim_ckpt, map_location='cpu', weights_only=True)
    victim.load_state_dict(ckpt, strict=True)
    victim.to(dev)

    attack = build_attack(args.attack, args)

    print("[RNP] initial evaluation:")
    ca0 = evaluate_clean_acc(victim, test_loader, dev)
    as0 = evaluate_asr(victim, test_loader, dev,
                       target_label=args.target, attack_obj=attack,
                       poison_source_classes=args.poison_source_classes)
    print(f"  Clean ACC: {ca0:.2f}%  ASR: {as0:.2f}%")

    # ------------------------------------------------------------------
    # Identify target layers
    # ------------------------------------------------------------------
    fc1_layers = _get_fc1_layers(victim, args.num_target_blocks)
    print(f"[RNP] targeting fc1 in last {len(fc1_layers)} blocks")

    # ------------------------------------------------------------------
    # Step 1: Collect baseline activations from original victim
    # ------------------------------------------------------------------
    print("[RNP] collecting baseline activations...")
    act_orig = collect_activations(victim, clean_loader, dev, fc1_layers)

    # ------------------------------------------------------------------
    # Step 2: Unlearn – maximize error on clean data (neuron-level)
    # ------------------------------------------------------------------
    model = copy.deepcopy(victim)
    fc1_layers_model = _get_fc1_layers(model, args.num_target_blocks)

    print(f"[RNP] unlearning for {args.unlearn_epochs} epochs "
          f"(lr={args.unlearn_lr})...")
    unlearn(model, clean_loader, dev,
            epochs=args.unlearn_epochs,
            lr=args.unlearn_lr,
            fc1_layers=fc1_layers_model)

    # ------------------------------------------------------------------
    # Step 3: Recover – minimize error on clean data
    # ------------------------------------------------------------------
    print(f"[RNP] recovering for {args.recover_epochs} epochs "
          f"(lr={args.recover_lr})...")
    recover(model, clean_loader, dev,
            epochs=args.recover_epochs,
            lr=args.recover_lr)

    # ------------------------------------------------------------------
    # Step 4: Compare activations and prune
    # ------------------------------------------------------------------
    print("[RNP] collecting recovered activations...")
    fc1_layers_model = _get_fc1_layers(model, args.num_target_blocks)
    act_recovered = collect_activations(model, clean_loader, dev,
                                        fc1_layers_model)

    print(f"[RNP] pruning top {args.prune_ratio*100:.0f}% neurons "
          f"by activation change...")
    n_pruned = prune_by_activation_change(model, act_orig, act_recovered,
                                          args.prune_ratio)
    print(f"[RNP] pruned {n_pruned} neurons")

    # ------------------------------------------------------------------
    # Step 5: Evaluate
    # ------------------------------------------------------------------
    caf = evaluate_clean_acc(model, test_loader, dev)
    asf = evaluate_asr(model, test_loader, dev,
                       target_label=args.target, attack_obj=attack,
                       poison_source_classes=args.poison_source_classes)
    print(f"[RNP] final: Clean={caf:.2f}% ASR={asf:.2f}%")
    print(f"\n[RESULT] RNP | {args.attack} | "
          f"clean={caf:.2f} ASR={asf:.2f}")


if __name__ == '__main__':
    main()
