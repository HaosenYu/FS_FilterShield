#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Neural Cleanse baseline defense.
Wang et al., "Neural Cleanse: Identifying and Mitigating Backdoor Attacks
in Neural Networks", S&P 2019.

Algorithm:
  1. For each class c, reverse-engineer a trigger (mask + pattern) that
     causes the model to predict c on any input, with minimal L1(mask).
     Solve:
        min_{m,p} E[L_ce(f((1-m)⊙x + m⊙p), c)] + λ * |m|_1
  2. Collect L1 norms across all classes, apply Median Absolute Deviation
     (MAD) anomaly detection. An outlier (very small mask) indicates the
     trigger target class.
  3. Mitigation: unlearning step — fine-tune the model on clean data
     with the reversed trigger attached but labeled correctly.

Note: NC is computationally expensive (K class optimizations). For 10-class
CIFAR-10 it's feasible; for 200-class Tiny-ImageNet it's very slow.

Usage:
  python baseline_neural_cleanse.py --victim_ckpt victim_badnets.pth \
    --attack badnets --trigger_size 4 \
    --data ./data --offline --no_download \
    --nc_epochs 10 --mitigation_epochs 5
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
    build_deit_tiny, build_vit_model, build_attack, build_transforms,
    build_dataset, NUM_CLASSES,
    evaluate_clean_acc, evaluate_asr,
    _assert_cifar10_present, _set_offline_env, set_seed, device,
    denorm, renorm,
)


def reverse_trigger_for_class(model, loader, target_class, dev,
                               mean, std, epochs=10, lr=0.1, init_cost=1e-3,
                               lambda_cap=100.0):
    """Reverse engineer a trigger for `target_class` using Neural Cleanse
    optimization. Returns (mask, pattern, l1_norm).

    mask: (1, H, W) in [0, 1]
    pattern: (C, H, W) in [0, 1]
    """
    img_size = 224
    C = 3
    mask_raw = torch.zeros(1, img_size, img_size, device=dev, requires_grad=True)
    pattern_raw = torch.zeros(C, img_size, img_size, device=dev, requires_grad=True)

    # Use tanh parameterization to keep in valid range
    def get_mask(): return torch.tanh(mask_raw) * 0.5 + 0.5  # [0,1]
    def get_pattern(): return torch.tanh(pattern_raw) * 0.5 + 0.5  # [0,1]

    opt = optim.Adam([mask_raw, pattern_raw], lr=lr, betas=(0.5, 0.9))

    cost = init_cost
    cost_multiplier = 1.5
    cost_down_rate = 0.7
    patience = 5
    asr_threshold = 0.99  # target: 99% attack success after trigger applied

    attack_success_count = 0
    attack_fail_count = 0

    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    for ep in range(epochs):
        total_ce = 0.0
        total_l1 = 0.0
        n_batch = 0
        correct_target = 0
        total = 0

        for x, y in loader:
            x = x.to(dev)
            # Convert to pixel space
            x_pix = denorm(x, mean, std).clamp(0, 1)

            m = get_mask()
            pat = get_pattern()
            x_trig = (1 - m) * x_pix + m * pat
            x_trig_norm = renorm(x_trig.clamp(0, 1), mean, std)

            logits = model(x_trig_norm)
            if isinstance(logits, tuple):
                logits = (logits[0] + logits[1]) / 2.0

            target_y = torch.full((x.size(0),), target_class,
                                  dtype=torch.long, device=dev)
            loss_ce = F.cross_entropy(logits, target_y)
            loss_l1 = m.abs().sum()
            loss = loss_ce + cost * loss_l1

            opt.zero_grad()
            loss.backward()
            opt.step()

            with torch.no_grad():
                pred = logits.argmax(dim=1)
                correct_target += (pred == target_class).sum().item()
                total += x.size(0)
                total_ce += loss_ce.item()
                total_l1 += loss_l1.item()
                n_batch += 1

        asr = correct_target / max(total, 1)
        avg_ce = total_ce / max(n_batch, 1)
        avg_l1 = total_l1 / max(n_batch, 1)

        # Adaptive cost scheduling
        if asr >= asr_threshold:
            attack_success_count += 1
            attack_fail_count = 0
            if attack_success_count >= patience:
                cost = min(cost * cost_multiplier, lambda_cap)
                attack_success_count = 0
        else:
            attack_fail_count += 1
            attack_success_count = 0
            if attack_fail_count >= patience:
                cost = max(cost * cost_down_rate, init_cost * 0.01)
                attack_fail_count = 0

        if ep % 2 == 0 or ep == epochs - 1:
            print(f"    class={target_class} ep={ep+1}/{epochs} "
                  f"asr_on_trig={asr:.3f} CE={avg_ce:.3f} "
                  f"L1={avg_l1:.1f} cost={cost:.4g}")

    with torch.no_grad():
        final_mask = get_mask().cpu()
        final_pat = get_pattern().cpu()
        l1 = final_mask.abs().sum().item()
    return final_mask, final_pat, l1


def mad_outlier_detection(values):
    """Median Absolute Deviation outlier detection.
    Returns anomaly index (lower is more anomalous)."""
    import numpy as np
    vals = torch.tensor(values)
    med = vals.median()
    mad = (vals - med).abs().median()
    if mad.item() == 0:
        mad = torch.tensor(1e-6)
    # Robust z-score
    z = (vals - med).abs() / (1.4826 * mad)
    return z.tolist(), med.item(), mad.item()


def mitigation_unlearn(model, loader, dev, mean, std, rev_mask, rev_pattern,
                       epochs, lr):
    """Unlearning: train the model with reversed trigger attached but with
    original (correct) labels, forcing the model to ignore the trigger."""
    # Re-enable gradients (reverse_trigger_for_class disabled them)
    for p in model.parameters():
        p.requires_grad_(True)
    model.train()
    opt = optim.Adam(model.parameters(), lr=lr)
    rev_mask = rev_mask.to(dev)
    rev_pattern = rev_pattern.to(dev)

    for ep in range(epochs):
        total = 0.0
        for x, y in loader:
            x, y = x.to(dev), y.to(dev)
            x_pix = denorm(x, mean, std).clamp(0, 1)
            x_trig = (1 - rev_mask) * x_pix + rev_mask * rev_pattern
            x_trig_norm = renorm(x_trig.clamp(0, 1), mean, std)

            out = model(x_trig_norm)
            if isinstance(out, tuple):
                out = (out[0] + out[1]) / 2.0
            loss = F.cross_entropy(out, y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item() * x.size(0)
        print(f"  [Unlearn] ep {ep+1}/{epochs} loss={total/len(loader.dataset):.4f}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--victim_ckpt', required=True)
    p.add_argument('--dataset', default='cifar10',
                   choices=['cifar10', 'gtsrb', 'tiny_imagenet'])
    p.add_argument('--model', default='deit_tiny',
                   choices=['deit_tiny', 'deit_small', 'deit_base',
                            'vit_b16', 'swin_t'])
    p.add_argument('--data', default='./data')
    p.add_argument('--offline', action='store_true')
    p.add_argument('--no_download', action='store_true')
    p.add_argument('--clean_fraction', type=float, default=0.05,
                   help='Fraction of train set for NC optimization + unlearning')
    p.add_argument('--batch_size', type=int, default=64)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--target', type=int, default=0)
    p.add_argument('--poison_source_classes', nargs='+', type=int, default=[3])
    # NC hyperparameters (--num_classes kept for backward compat but overridden by --dataset)
    p.add_argument('--num_classes', type=int, default=None)
    p.add_argument('--nc_epochs', type=int, default=10)
    p.add_argument('--nc_lr', type=float, default=0.1)
    p.add_argument('--nc_init_cost', type=float, default=1e-3)
    p.add_argument('--mad_threshold', type=float, default=2.0,
                   help='MAD z-score threshold for outlier detection')
    p.add_argument('--mitigation_epochs', type=int, default=5)
    p.add_argument('--mitigation_lr', type=float, default=1e-4)
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

    # Resolve num_classes: explicit flag > dataset lookup
    num_classes = args.num_classes if args.num_classes is not None else NUM_CLASSES[args.dataset]

    if args.offline:
        _set_offline_env()
        if args.dataset == 'cifar10':
            _assert_cifar10_present(args.data)
    set_seed(args.seed)
    dev = device()

    train_tf, test_tf, mean, std = build_transforms(img_size=224, dataset=args.dataset)
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

    victim = build_vit_model(args.model, pretrained=False, num_classes=num_classes).to(dev)
    ckpt = torch.load(args.victim_ckpt, map_location='cpu', weights_only=True)
    victim.load_state_dict(ckpt, strict=True)
    victim.to(dev)

    attack = build_attack(args.attack, args)

    print("[NC] initial:")
    ca0 = evaluate_clean_acc(victim, test_loader, dev)
    as0 = evaluate_asr(victim, test_loader, dev,
                       target_label=args.target, attack_obj=attack,
                       poison_source_classes=args.poison_source_classes)
    print(f"  Clean ACC: {ca0:.2f}%  ASR: {as0:.2f}%")

    # Step 1: Reverse-engineer trigger for each class
    print(f"[NC] reverse-engineering triggers for {num_classes} classes...")
    l1_norms = []
    masks = []
    patterns = []
    for c in range(num_classes):
        print(f"  === class {c}/{num_classes - 1} ===")
        m, pat, l1 = reverse_trigger_for_class(
            victim, clean_loader, c, dev, mean, std,
            epochs=args.nc_epochs, lr=args.nc_lr,
            init_cost=args.nc_init_cost,
        )
        l1_norms.append(l1)
        masks.append(m)
        patterns.append(pat)
        print(f"  class {c}: L1 = {l1:.1f}")

    # Step 2: Anomaly detection
    zs, med, mad_val = mad_outlier_detection(l1_norms)
    print(f"\n[NC] L1 norms: {[f'{v:.1f}' for v in l1_norms]}")
    print(f"[NC] Median: {med:.1f}, MAD: {mad_val:.1f}")
    print(f"[NC] z-scores: {[f'{z:.2f}' for z in zs]}")

    # Outlier = class with small L1 (easy to trigger → suspicious)
    min_idx = int(torch.tensor(l1_norms).argmin().item())
    suspected = min_idx if zs[min_idx] > args.mad_threshold else None

    print(f"[NC] Smallest L1 at class {min_idx}, z={zs[min_idx]:.2f}, "
          f"threshold={args.mad_threshold}")
    if suspected is None:
        print("[NC] WARNING: no class flagged as backdoor target. "
              "Using smallest-L1 class for mitigation.")
        suspected = min_idx
    print(f"[NC] Suspected target class: {suspected} "
          f"(true target: {args.target})")

    # Step 3: Unlearn using the reversed trigger
    print(f"[NC] unlearning with reversed trigger from class {suspected}...")
    model = copy.deepcopy(victim)
    mitigation_unlearn(model, clean_loader, dev, mean, std,
                       masks[suspected], patterns[suspected],
                       epochs=args.mitigation_epochs, lr=args.mitigation_lr)

    caf = evaluate_clean_acc(model, test_loader, dev)
    asf = evaluate_asr(model, test_loader, dev,
                       target_label=args.target, attack_obj=attack,
                       poison_source_classes=args.poison_source_classes)
    print(f"[NC] final: Clean={caf:.2f}% ASR={asf:.2f}%")
    print(f"\n[RESULT] NC | {args.attack} | "
          f"Clean: {ca0:.2f}→{caf:.2f}  ASR: {as0:.2f}→{asf:.2f} "
          f"| detected={suspected} true={args.target}")


if __name__ == '__main__':
    main()
