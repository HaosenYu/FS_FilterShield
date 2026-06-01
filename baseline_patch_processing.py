#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Patch Processing baseline defense (test-time only).
Doan et al., "Defending Backdoor Attacks on Vision Transformer via Patch
Processing," AAAI 2023.

Key idea: ViTs are sensitive to patch-level transformations. By applying
random perturbations (local smoothing, patch dropout, patch shuffling) to
image patches before inference, backdoor triggers are disrupted.

Usage:
  python baseline_patch_processing.py \
    --victim_ckpt victim_badnets.pth \
    --attack badnets --trigger_size 4 \
    --data ./data --offline --no_download \
    --patch_drop_prob 0.1 --smooth_sigma 1.0 --num_votes 1
"""
import argparse, os, sys
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from deit_tiny_fs_plus_offline import (
    build_vit_model, build_attack, build_transforms, build_dataset,
    NUM_CLASSES, evaluate_clean_acc, evaluate_asr,
    _assert_cifar10_present, _set_offline_env, set_seed, device,
)


# ── patch-level perturbations ───────────────────────────────────────────

def patch_perturb(images: torch.Tensor, patch_size: int = 16,
                  drop_prob: float = 0.1, shuffle_prob: float = 0.0,
                  smooth_sigma: float = 1.0) -> torch.Tensor:
    """Apply patch-level perturbations to a batch of images.

    Args:
        images: (B, C, H, W) tensor.
        patch_size: size of each square patch (must divide H and W).
        drop_prob: probability of zeroing out each patch.
        shuffle_prob: probability of randomly shuffling patch positions.
        smooth_sigma: std-dev of Gaussian smoothing applied per patch
                      (0 = no smoothing).
    Returns:
        Perturbed image tensor of the same shape.
    """
    B, C, H, W = images.shape
    nph, npw = H // patch_size, W // patch_size
    x = images.clone()

    # ---------- local Gaussian smoothing per patch ----------
    if smooth_sigma > 0:
        ks = max(3, int(2 * round(smooth_sigma) + 1))
        if ks % 2 == 0:
            ks += 1
        padding = ks // 2
        # Build 1-D Gaussian kernel
        coord = torch.arange(ks, dtype=x.dtype, device=x.device) - ks // 2
        g1d = torch.exp(-0.5 * (coord / smooth_sigma) ** 2)
        g1d = g1d / g1d.sum()
        kernel = g1d[:, None] * g1d[None, :]              # (ks, ks)
        kernel = kernel.expand(C, 1, ks, ks)               # depthwise
        x = F.conv2d(x, kernel, padding=padding, groups=C)

    # ---------- patch dropout ----------
    if drop_prob > 0:
        mask = torch.rand(B, 1, nph, npw, device=x.device) > drop_prob
        mask = mask.float()
        mask = mask.repeat_interleave(patch_size, dim=2) \
                    .repeat_interleave(patch_size, dim=3)
        x = x * mask

    # ---------- patch shuffle ----------
    if shuffle_prob > 0 and torch.rand(1).item() < shuffle_prob:
        # reshape into patches, shuffle along patch-sequence dim, reshape back
        x = x.view(B, C, nph, patch_size, npw, patch_size)
        x = x.permute(0, 1, 2, 4, 3, 5).contiguous()     # (B, C, nph, npw, p, p)
        x = x.view(B, C, nph * npw, patch_size, patch_size)
        idx = torch.randperm(nph * npw, device=x.device).expand(B, C, -1)
        idx = idx.unsqueeze(-1).unsqueeze(-1).expand_as(x)
        x = torch.gather(x, 2, idx)
        x = x.view(B, C, nph, npw, patch_size, patch_size)
        x = x.permute(0, 1, 2, 4, 3, 5).contiguous()
        x = x.view(B, C, H, W)

    return x


# ── evaluation helpers ──────────────────────────────────────────────────

@torch.no_grad()
def eval_with_patch_defense(model, loader, dev, args, attack_obj=None,
                            target_label=None, poison_source_classes=None):
    """Evaluate clean ACC or ASR under patch-processing defense.

    If attack_obj is None, evaluates clean accuracy.
    If attack_obj is provided, evaluates ASR (applies attack first, then
    patch defense).
    """
    model.eval()
    correct, total = 0, 0
    for images, labels in loader:
        images, labels = images.to(dev), labels.to(dev)

        if attack_obj is not None:
            # Filter to source classes only
            if poison_source_classes is not None:
                mask = torch.zeros(len(labels), dtype=torch.bool, device=dev)
                for sc in poison_source_classes:
                    mask |= (labels == sc)
                if mask.sum() == 0:
                    continue
                images, labels = images[mask], labels[mask]
            images = attack_obj.apply(images)
            labels = torch.full_like(labels, target_label)

        # Majority vote over K stochastic runs
        votes = []
        for _ in range(args.num_votes):
            x_p = patch_perturb(images, patch_size=args.patch_size,
                                drop_prob=args.patch_drop_prob,
                                shuffle_prob=args.patch_shuffle_prob,
                                smooth_sigma=args.smooth_sigma)
            out = model(x_p)
            if isinstance(out, tuple):
                out = (out[0] + out[1]) / 2.0
            votes.append(out.argmax(dim=1))

        if args.num_votes == 1:
            preds = votes[0]
        else:
            stacked = torch.stack(votes, dim=0)            # (K, B)
            preds = torch.mode(stacked, dim=0).values

        correct += (preds == labels).sum().item()
        total += labels.size(0)

    return 100.0 * correct / max(total, 1)


# ── main ────────────────────────────────────────────────────────────────

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
    p.add_argument('--batch_size', type=int, default=128)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--target', type=int, default=0)
    p.add_argument('--poison_source_classes', nargs='+', type=int, default=[3])
    # Defense hyperparameters
    p.add_argument('--patch_size', type=int, default=16)
    p.add_argument('--patch_drop_prob', type=float, default=0.1)
    p.add_argument('--patch_shuffle_prob', type=float, default=0.0)
    p.add_argument('--smooth_sigma', type=float, default=1.0)
    p.add_argument('--num_votes', type=int, default=1)
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

    _, test_tf, _, _ = build_transforms(img_size=224, dataset=dataset_name)
    download = (not args.offline) and (not args.no_download)
    test_ds = build_dataset(dataset_name, args.data, transform=test_tf,
                            train=False, download=download)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size,
                             shuffle=False, num_workers=4, pin_memory=True)

    # Load victim model
    victim = build_vit_model(model_name, pretrained=False,
                             num_classes=num_cls).to(dev)
    ckpt = torch.load(args.victim_ckpt, map_location='cpu', weights_only=True)
    victim.load_state_dict(ckpt, strict=True)
    victim.to(dev)
    victim.eval()

    attack = build_attack(args.attack, args)

    print(f"[PatchProcess] patch_size={args.patch_size} "
          f"drop={args.patch_drop_prob} shuffle={args.patch_shuffle_prob} "
          f"sigma={args.smooth_sigma} votes={args.num_votes}")

    clean_acc = eval_with_patch_defense(victim, test_loader, dev, args)
    asr = eval_with_patch_defense(victim, test_loader, dev, args,
                                  attack_obj=attack,
                                  target_label=args.target,
                                  poison_source_classes=args.poison_source_classes)

    print(f"  Clean ACC: {clean_acc:.2f}%  ASR: {asr:.2f}%")
    print(f"\n[RESULT] PatchProcess | {args.attack} | "
          f"clean={clean_acc:.2f} ASR={asr:.2f}")


if __name__ == '__main__':
    main()
