#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fine-Pruning baseline defense (Liu et al., RAID 2018) adapted for ViT family.

Strategy:
  1. Load victim model.
  2. Forward a small clean subset, record average activation magnitude
     per neuron in the last transformer block's MLP (fc1 output).
  3. Prune the N% least-activated neurons by zeroing fc1 rows and fc2 cols.
  4. Fine-tune on the clean subset for a few epochs.
  5. Evaluate Clean ACC and ASR.

Usage:
  python baseline_finepruning.py \
    --victim_ckpt victim_badnets.pth \
    --attack badnets --trigger_size 4 \
    --data ./data --offline --pretrained_ckpt ./weights/deit_tiny_pretrained.pth \
    --prune_ratio 0.3 --ft_epochs 5 \
    --dataset cifar10 --model deit_tiny
"""
import argparse, os, sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
import torchvision

# Import shared utilities from the main FS script (must be in the same directory).
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from deit_tiny_fs_plus_offline import (
    build_deit_tiny, build_vit_model, build_attack, build_transforms,
    build_dataset, NUM_CLASSES, get_transformer_blocks,
    evaluate_clean_acc, evaluate_asr,
    _assert_cifar10_present, _set_offline_env, set_seed, device,
)


def collect_activations(model, loader, dev, block_idx=-1):
    """Record mean |activation| per neuron in block blocks[block_idx].mlp.fc1 output."""
    blocks = get_transformer_blocks(model)
    block = blocks[block_idx]
    fc1 = block.mlp.fc1
    hidden_dim = fc1.out_features

    sum_abs = torch.zeros(hidden_dim, device=dev)
    count = 0

    hook_out = {}

    def hook(_mod, _inp, out):
        hook_out['x'] = out

    handle = fc1.register_forward_hook(hook)
    model.eval()
    with torch.no_grad():
        for x, _y in loader:
            x = x.to(dev)
            _ = model(x)
            act = hook_out['x']  # (B, tokens, hidden)
            sum_abs += act.abs().mean(dim=(0, 1))
            count += 1
    handle.remove()
    return (sum_abs / max(count, 1)).cpu()


def prune_neurons(model, importance, prune_ratio, block_idx=-1):
    """Zero out the bottom `prune_ratio` neurons by importance."""
    blocks = get_transformer_blocks(model)
    block = blocks[block_idx]
    fc1, fc2 = block.mlp.fc1, block.mlp.fc2
    hidden = importance.numel()
    n_prune = int(hidden * prune_ratio)
    if n_prune <= 0:
        return 0
    _, idx_sorted = torch.sort(importance)
    prune_idx = idx_sorted[:n_prune].tolist()

    with torch.no_grad():
        # fc1: (hidden, in) -> zero rows
        fc1.weight[prune_idx, :] = 0.0
        if fc1.bias is not None:
            fc1.bias[prune_idx] = 0.0
        # fc2: (out, hidden) -> zero columns
        fc2.weight[:, prune_idx] = 0.0
    return n_prune


def fine_tune(model, loader, dev, epochs=5, lr=1e-4):
    model.train()
    opt = optim.Adam(model.parameters(), lr=lr)
    for ep in range(epochs):
        total = 0.0
        for x, y in loader:
            x, y = x.to(dev), y.to(dev)
            out = model(x)
            if isinstance(out, tuple):
                out = (out[0] + out[1]) / 2.0
            loss = F.cross_entropy(out, y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item() * x.size(0)
        print(f"  [FineTune] epoch {ep + 1}/{epochs} loss={total / len(loader.dataset):.4f}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--victim_ckpt', required=True)
    p.add_argument('--data', default='./data')
    p.add_argument('--dataset', default='cifar10',
                   choices=['cifar10', 'gtsrb', 'tiny_imagenet'])
    p.add_argument('--model', default='deit_tiny',
                   choices=['deit_tiny', 'deit_small', 'deit_base', 'vit_b16', 'swin_t'])
    p.add_argument('--offline', action='store_true')
    p.add_argument('--no_download', action='store_true')
    p.add_argument('--pretrained_ckpt', default='')
    p.add_argument('--prune_ratio', type=float, default=0.3)
    p.add_argument('--ft_epochs', type=int, default=5)
    p.add_argument('--ft_lr', type=float, default=1e-4)
    p.add_argument('--clean_fraction', type=float, default=0.10)
    p.add_argument('--batch_size', type=int, default=128)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--target', type=int, default=0)
    p.add_argument('--poison_source_classes', nargs='+', type=int, default=[3])
    # Attack args (subset)
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

    if args.offline:
        _set_offline_env()
        if args.dataset == 'cifar10':
            _assert_cifar10_present(args.data)

    set_seed(args.seed)
    dev = device()
    num_classes = NUM_CLASSES[args.dataset]

    train_tf, test_tf, _, _ = build_transforms(img_size=224, dataset=args.dataset)
    download = (not args.offline) and (not args.no_download)
    full_train = build_dataset(args.dataset, args.data, train_tf,
                               train=True, download=download)
    test_ds = build_dataset(args.dataset, args.data, test_tf,
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

    # Load victim
    model = build_vit_model(args.model, pretrained=False,
                            num_classes=num_classes).to(dev)
    ckpt = torch.load(args.victim_ckpt, map_location='cpu', weights_only=True)
    model.load_state_dict(ckpt, strict=True)
    model.to(dev)

    attack = build_attack(args.attack, args)

    print("[FinePruning] initial:")
    clean_acc_0 = evaluate_clean_acc(model, test_loader, dev)
    asr_0 = evaluate_asr(model, test_loader, dev,
                         target_label=args.target, attack_obj=attack,
                         poison_source_classes=args.poison_source_classes)
    print(f"  Clean ACC: {clean_acc_0:.2f}%  ASR: {asr_0:.2f}%")

    print(f"[FinePruning] collecting activations on {k} clean samples...")
    importance = collect_activations(model, clean_loader, dev, block_idx=-1)
    print(f"  importance range: [{importance.min():.4f}, {importance.max():.4f}]")

    n_pruned = prune_neurons(model, importance, args.prune_ratio, block_idx=-1)
    print(f"[FinePruning] pruned {n_pruned} neurons ({args.prune_ratio*100:.0f}%)")

    clean_acc_p = evaluate_clean_acc(model, test_loader, dev)
    asr_p = evaluate_asr(model, test_loader, dev,
                         target_label=args.target, attack_obj=attack,
                         poison_source_classes=args.poison_source_classes)
    print(f"[FinePruning] after pruning:")
    print(f"  Clean ACC: {clean_acc_p:.2f}%  ASR: {asr_p:.2f}%")

    print(f"[FinePruning] fine-tuning {args.ft_epochs} epochs...")
    fine_tune(model, clean_loader, dev, epochs=args.ft_epochs, lr=args.ft_lr)

    clean_acc_f = evaluate_clean_acc(model, test_loader, dev)
    asr_f = evaluate_asr(model, test_loader, dev,
                         target_label=args.target, attack_obj=attack,
                         poison_source_classes=args.poison_source_classes)
    print(f"[FinePruning] final:")
    print(f"  Clean ACC: {clean_acc_f:.2f}%  ASR: {asr_f:.2f}%")

    print(f"\n[RESULT] {args.attack} | prune_ratio={args.prune_ratio} | "
          f"Clean: {clean_acc_0:.2f}→{clean_acc_f:.2f}  ASR: {asr_0:.2f}→{asr_f:.2f}")


if __name__ == '__main__':
    main()
