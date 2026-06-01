#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
NAD (Neural Attention Distillation) baseline defense.
Li et al. "Neural Attention Distillation: Erasing Backdoor Triggers from
Deep Neural Networks", ICLR 2021.

Algorithm:
  1. Copy the backdoored (victim) model to create a 'teacher'.
  2. Fine-tune the teacher for a few epochs on a small clean subset.
  3. Use the teacher to distill attention back into the 'student' (another
     copy of the victim), forcing the student's attention maps to match
     the teacher's at intermediate layers.
  4. The resulting student has reduced backdoor behavior.

For ViTs, we use the output of each transformer block (mean over tokens)
as the "attention feature map". The distillation loss is the L2 distance
between normalized feature maps.

Usage:
  python baseline_nad.py \
    --victim_ckpt victim_badnets.pth \
    --attack badnets --trigger_size 4 \
    --data ./data --offline --no_download \
    --teacher_epochs 5 --distill_epochs 10 --beta 500
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


class FeatureCollector:
    """Hook on each transformer block to record output features."""
    def __init__(self, model):
        self.feats = []
        self.handles = []
        for blk in get_transformer_blocks(model):
            h = blk.register_forward_hook(self._hook)
            self.handles.append(h)

    def _hook(self, _mod, _inp, out):
        # out: (B, tokens, dim)
        self.feats.append(out)

    def clear(self):
        self.feats = []

    def remove(self):
        for h in self.handles:
            h.remove()


def at_map(feat: torch.Tensor) -> torch.Tensor:
    """Attention feature map from block output.
    For ViT: take the L2 norm over feature dim per token, then normalize.
    Shape: (B, tokens)."""
    att = feat.pow(2).sum(dim=-1)  # (B, tokens)
    att = F.normalize(att, dim=1, p=2)
    return att


def distillation_loss(student_feats, teacher_feats):
    """Sum of L2 losses between attention maps at each block."""
    loss = 0.0
    n = min(len(student_feats), len(teacher_feats))
    for i in range(n):
        s = at_map(student_feats[i])
        t = at_map(teacher_feats[i]).detach()
        loss = loss + (s - t).pow(2).mean()
    return loss


def fine_tune_teacher(teacher, loader, dev, epochs, lr):
    teacher.train()
    opt = optim.Adam(teacher.parameters(), lr=lr)
    for ep in range(epochs):
        total = 0.0
        for x, y in loader:
            x, y = x.to(dev), y.to(dev)
            out = teacher(x)
            if isinstance(out, tuple):
                out = (out[0] + out[1]) / 2.0
            loss = F.cross_entropy(out, y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item() * x.size(0)
        print(f"  [Teacher] epoch {ep + 1}/{epochs} loss={total / len(loader.dataset):.4f}")


def distill(student, teacher, loader, dev, epochs, lr, beta):
    student.train()
    teacher.eval()
    opt = optim.Adam(student.parameters(), lr=lr)

    s_hook = FeatureCollector(student)
    t_hook = FeatureCollector(teacher)

    try:
        for ep in range(epochs):
            running_ce = 0.0
            running_at = 0.0
            for x, y in loader:
                x, y = x.to(dev), y.to(dev)

                s_hook.clear()
                t_hook.clear()

                s_out = student(x)
                if isinstance(s_out, tuple):
                    s_logits = (s_out[0] + s_out[1]) / 2.0
                else:
                    s_logits = s_out

                with torch.no_grad():
                    _ = teacher(x)

                loss_ce = F.cross_entropy(s_logits, y)
                loss_at = distillation_loss(s_hook.feats, t_hook.feats)
                loss = loss_ce + beta * loss_at

                opt.zero_grad()
                loss.backward()
                opt.step()

                running_ce += loss_ce.item() * x.size(0)
                running_at += loss_at.item() * x.size(0)
            n = len(loader.dataset)
            print(f"  [Distill] epoch {ep + 1}/{epochs} "
                  f"CE={running_ce/n:.4f} AT={running_at/n:.6f}")
    finally:
        s_hook.remove()
        t_hook.remove()


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
    # NAD hyperparameters
    p.add_argument('--teacher_epochs', type=int, default=5)
    p.add_argument('--teacher_lr', type=float, default=1e-4)
    p.add_argument('--distill_epochs', type=int, default=10)
    p.add_argument('--distill_lr', type=float, default=1e-4)
    p.add_argument('--beta', type=float, default=500.0,
                   help='Weight of attention distillation loss')
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

    if args.offline:
        _set_offline_env()
        _assert_cifar10_present(args.data)
    set_seed(args.seed)
    dev = device()

    dataset_name = getattr(args, 'dataset', 'cifar10')
    model_name = getattr(args, 'model', 'deit_tiny')
    num_cls = NUM_CLASSES.get(dataset_name, 10)

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

    # Load victim
    victim = build_vit_model(model_name, pretrained=False, num_classes=num_cls).to(dev)
    ckpt = torch.load(args.victim_ckpt, map_location='cpu', weights_only=True)
    victim.load_state_dict(ckpt, strict=True)
    victim.to(dev)

    attack = build_attack(args.attack, args)

    print("[NAD] initial:")
    ca0 = evaluate_clean_acc(victim, test_loader, dev)
    as0 = evaluate_asr(victim, test_loader, dev,
                       target_label=args.target, attack_obj=attack,
                       poison_source_classes=args.poison_source_classes)
    print(f"  Clean ACC: {ca0:.2f}%  ASR: {as0:.2f}%")

    # Build teacher (fine-tuned copy)
    print(f"[NAD] fine-tuning teacher on {k} clean samples, "
          f"{args.teacher_epochs} epochs...")
    teacher = copy.deepcopy(victim)
    fine_tune_teacher(teacher, clean_loader, dev,
                      epochs=args.teacher_epochs, lr=args.teacher_lr)

    # Distill into student (another copy of victim)
    print(f"[NAD] distilling student from teacher, "
          f"{args.distill_epochs} epochs, beta={args.beta}...")
    student = copy.deepcopy(victim)
    distill(student, teacher, clean_loader, dev,
            epochs=args.distill_epochs, lr=args.distill_lr, beta=args.beta)

    caf = evaluate_clean_acc(student, test_loader, dev)
    asf = evaluate_asr(student, test_loader, dev,
                       target_label=args.target, attack_obj=attack,
                       poison_source_classes=args.poison_source_classes)
    print(f"[NAD] final (student):")
    print(f"  Clean ACC: {caf:.2f}%  ASR: {asf:.2f}%")
    print(f"\n[RESULT] NAD | {args.attack} | "
          f"Clean: {ca0:.2f}→{caf:.2f}  ASR: {as0:.2f}→{asf:.2f}")


if __name__ == '__main__':
    main()
