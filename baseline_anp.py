#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ANP (Adversarial Neuron Pruning) baseline defense.
Wu & Wang, "Adversarial Neuron Pruning Purifies Backdoored Deep Models",
NeurIPS 2021.

Algorithm:
  1. Attach learnable per-neuron masks m ∈ [0,1] to selected layers
     (we target the MLP fc1 outputs in each transformer block).
  2. Optimize masks m and adversarial perturbations δ on the mask jointly:
        min_m max_{||δ||≤ε} E[L_ce(f(x; m ⊙ (1 + δ)), y)]
     where δ perturbs the mask. Neurons that become "sensitive" under δ
     are assumed backdoor-related.
  3. After optimization, prune neurons with mask value < threshold.
  4. Optional: fine-tune on clean subset.

For DeiT-tiny, we attach masks to the output of MLP.fc1 in each block.

Usage:
  python baseline_anp.py --victim_ckpt victim_badnets.pth \
    --attack badnets --trigger_size 4 \
    --data ./data --offline --no_download \
    --anp_epochs 20 --mask_threshold 0.4
"""
import argparse, os, sys, copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
import torchvision

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from deit_tiny_fs_plus_offline import (
    build_vit_model, build_attack, build_transforms, build_dataset,
    NUM_CLASSES, evaluate_clean_acc, evaluate_asr,
    _assert_cifar10_present, _set_offline_env, set_seed, device,
    get_transformer_blocks,
)


class NeuronMask(nn.Module):
    """Wraps a Linear layer with a learnable per-output-neuron mask and
    an adversarial perturbation on the mask."""
    def __init__(self, linear: nn.Linear, eps: float = 0.4):
        super().__init__()
        self.linear = linear
        self.eps = eps
        hidden = linear.out_features
        # Create parameters on the same device as the wrapped linear layer
        dev = linear.weight.device
        self.mask = nn.Parameter(torch.ones(hidden, device=dev))
        self.delta = nn.Parameter(torch.zeros(hidden, device=dev))
        self.training_mask = True  # switch between training mode

    def forward(self, x):
        out = self.linear(x)
        # effective mask = mask * (1 + delta), clamped to [0, 1]
        eff = torch.clamp(self.mask * (1.0 + self.delta), 0.0, 1.0)
        return out * eff.view(1, 1, -1) if out.dim() == 3 else out * eff.view(1, -1)


def install_masks(model, eps):
    """Replace each block's mlp.fc1 with a NeuronMask wrapper."""
    masks = []
    blocks = get_transformer_blocks(model)
    for i, blk in enumerate(blocks):
        fc1 = blk.mlp.fc1
        wrapper = NeuronMask(fc1, eps=eps)
        blk.mlp.fc1 = wrapper
        masks.append((i, wrapper))
    return masks


def remove_masks_and_prune(model, masks, threshold):
    """Replace wrappers back with Linear, zeroing out pruned neurons."""
    n_pruned_total = 0
    blocks = get_transformer_blocks(model)
    for i, wrapper in masks:
        with torch.no_grad():
            m = wrapper.mask.detach().clamp(0, 1)
            prune_mask = (m < threshold).float()
            keep_mask = 1.0 - prune_mask
            n_pruned = int(prune_mask.sum().item())
            n_pruned_total += n_pruned

            fc1 = wrapper.linear
            fc2 = blocks[i].mlp.fc2
            fc1.weight.data = fc1.weight.data * keep_mask.view(-1, 1)
            if fc1.bias is not None:
                fc1.bias.data = fc1.bias.data * keep_mask
            fc2.weight.data = fc2.weight.data * keep_mask.view(1, -1)

        blocks[i].mlp.fc1 = fc1  # restore original
    return n_pruned_total


def anp_optimize(model, masks, loader, dev, epochs, lr_mask, lr_delta):
    """Alternating optimization:
       - Step A: update delta to maximize CE loss (adversarial step)
       - Step B: update mask to minimize CE loss under the perturbed mask
    """
    mask_params = [w.mask for _, w in masks]
    delta_params = [w.delta for _, w in masks]
    other_params = [p for n, p in model.named_parameters()
                    if not any(n.endswith(k) for k in ['mask', 'delta'])]
    for p in other_params:
        p.requires_grad_(False)

    opt_mask = optim.SGD(mask_params, lr=lr_mask, momentum=0.9)
    opt_delta = optim.SGD(delta_params, lr=lr_delta, momentum=0.9)

    model.train()
    for ep in range(epochs):
        total = 0.0
        for x, y in loader:
            x, y = x.to(dev), y.to(dev)

            # Step A: maximize loss over delta (adversarial)
            opt_delta.zero_grad()
            out = model(x)
            if isinstance(out, tuple):
                out = (out[0] + out[1]) / 2.0
            loss_adv = -F.cross_entropy(out, y)  # negate → ascent
            loss_adv.backward()
            opt_delta.step()
            with torch.no_grad():
                for _, w in masks:
                    w.delta.clamp_(-w.eps, w.eps)

            # Step B: minimize loss over mask (under current delta)
            opt_mask.zero_grad()
            out = model(x)
            if isinstance(out, tuple):
                out = (out[0] + out[1]) / 2.0
            loss = F.cross_entropy(out, y)
            loss.backward()
            opt_mask.step()
            with torch.no_grad():
                for _, w in masks:
                    w.mask.clamp_(0.0, 1.0)

            total += loss.item() * x.size(0)

        # Print mask stats
        all_masks = torch.cat([w.mask.detach() for _, w in masks])
        print(f"  [ANP] ep {ep+1}/{epochs} loss={total/len(loader.dataset):.4f} "
              f"mask mean={all_masks.mean():.3f} "
              f"min={all_masks.min():.3f} "
              f"frac<0.3={(all_masks<0.3).float().mean():.3f}")


def fine_tune(model, loader, dev, epochs, lr):
    for p in model.parameters():
        p.requires_grad_(True)
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
        print(f"  [FT] ep {ep+1}/{epochs} loss={total/len(loader.dataset):.4f}")


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
    p.add_argument('--clean_fraction', type=float, default=0.10)
    p.add_argument('--batch_size', type=int, default=128)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--target', type=int, default=0)
    p.add_argument('--poison_source_classes', nargs='+', type=int, default=[3])
    # ANP hyperparameters
    p.add_argument('--anp_epochs', type=int, default=20)
    p.add_argument('--lr_mask', type=float, default=0.2)
    p.add_argument('--lr_delta', type=float, default=0.2)
    p.add_argument('--anp_eps', type=float, default=0.4)
    p.add_argument('--mask_threshold', type=float, default=0.4)
    p.add_argument('--ft_epochs', type=int, default=3)
    p.add_argument('--ft_lr', type=float, default=1e-4)
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

    train_tf, test_tf, _, _ = build_transforms(img_size=224, dataset=dataset_name)
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

    victim = build_vit_model(model_name, pretrained=False, num_classes=num_cls).to(dev)
    ckpt = torch.load(args.victim_ckpt, map_location='cpu', weights_only=True)
    victim.load_state_dict(ckpt, strict=True)
    victim.to(dev)

    attack = build_attack(args.attack, args)

    print("[ANP] initial:")
    ca0 = evaluate_clean_acc(victim, test_loader, dev)
    as0 = evaluate_asr(victim, test_loader, dev,
                       target_label=args.target, attack_obj=attack,
                       poison_source_classes=args.poison_source_classes)
    print(f"  Clean ACC: {ca0:.2f}%  ASR: {as0:.2f}%")

    # Install masks
    model = copy.deepcopy(victim)
    masks = install_masks(model, eps=args.anp_eps)
    print(f"[ANP] installed masks on {len(masks)} blocks")

    # Optimize
    anp_optimize(model, masks, clean_loader, dev,
                 epochs=args.anp_epochs,
                 lr_mask=args.lr_mask, lr_delta=args.lr_delta)

    # Prune
    n_pruned = remove_masks_and_prune(model, masks, threshold=args.mask_threshold)
    print(f"[ANP] pruned {n_pruned} neurons with mask < {args.mask_threshold}")

    # Evaluate before fine-tuning
    cap = evaluate_clean_acc(model, test_loader, dev)
    asp = evaluate_asr(model, test_loader, dev,
                       target_label=args.target, attack_obj=attack,
                       poison_source_classes=args.poison_source_classes)
    print(f"[ANP] after pruning: Clean={cap:.2f}% ASR={asp:.2f}%")

    # Optional fine-tune
    if args.ft_epochs > 0:
        print(f"[ANP] fine-tuning {args.ft_epochs} epochs...")
        fine_tune(model, clean_loader, dev, epochs=args.ft_epochs, lr=args.ft_lr)

    caf = evaluate_clean_acc(model, test_loader, dev)
    asf = evaluate_asr(model, test_loader, dev,
                       target_label=args.target, attack_obj=attack,
                       poison_source_classes=args.poison_source_classes)
    print(f"[ANP] final: Clean={caf:.2f}% ASR={asf:.2f}%")
    print(f"\n[RESULT] ANP | {args.attack} | "
          f"Clean: {ca0:.2f}→{caf:.2f}  ASR: {as0:.2f}→{asf:.2f}")


if __name__ == '__main__':
    main()
