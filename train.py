"""
ASL SIGN RECOGNITION TRAINER — v4
==================================

Changes over v3:
1.  Top-N classes by signer diversity   : 109 clips/class avg vs 44 before
2.  Imbalance ratio 142x → 24x         : weighted sampler far more effective now
3.  num_classes 381 → 100              : tractable problem for available data + GPU
4.  GTX 1650 safe defaults             : 160px, 8 frames, batch 4
"""

import os
import cv2
import time
import random
import logging
import warnings
import argparse

import numpy as np
import pandas as pd

from pathlib import Path
from collections import Counter
from typing import List

import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import Dataset, DataLoader, Subset, WeightedRandomSampler
from sklearn.model_selection import train_test_split

from torchvision.models.video import r3d_18, R3D_18_Weights

from tqdm import tqdm

warnings.filterwarnings("ignore")


# CONFIG

def get_config():
    parser = argparse.ArgumentParser()
    parser.add_argument("--excel",           default=r"D:\aslp\data\asl_windowed_dataset_cleaned.xlsx")
    parser.add_argument("--video_root",      default=r"D:\aslp\data\final_videos")
    parser.add_argument("--ckpt_dir",        default=r"D:\aslp\checkpoints_v4")
    parser.add_argument("--img_size",        type=int,   default=160)  
    parser.add_argument("--num_frames",      type=int,   default=8)     
    parser.add_argument("--batch_size",      type=int,   default=4)     
    parser.add_argument("--num_workers",     type=int,   default=4)
    parser.add_argument("--val_split",       type=float, default=0.10)
    parser.add_argument("--test_split",      type=float, default=0.10)
    parser.add_argument("--epochs",          type=int,   default=40)
    parser.add_argument("--freeze_epochs",   type=int,   default=5)
    parser.add_argument("--lr_frozen",       type=float, default=1e-4)
    parser.add_argument("--lr_unfrozen",     type=float, default=3e-5)
    parser.add_argument("--weight_decay",    type=float, default=1e-4)
    parser.add_argument("--dropout",         type=float, default=0.5)
    parser.add_argument("--grad_clip",       type=float, default=1.0)
    parser.add_argument("--seed",            type=int,   default=42)
    parser.add_argument("--top_n_classes",   type=int,   default=100)  
    parser.add_argument("--label_smoothing", type=float, default=0.1)
    return parser.parse_args()


# LOGGING

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")
log = logging.getLogger("ASL-v4")


# SEED

def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark        = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32       = True


# FRAME SAMPLING

def sample_indices(total_frames: int, num_frames: int, augment: bool) -> List[int]:
    """
    Divide video into num_frames equal segments.
    Train   : random frame per segment (temporal jitter)
    Val/Test: center frame per segment (deterministic)
    """
    segment_size = total_frames / num_frames
    indices = []
    for i in range(num_frames):
        start = int(i * segment_size)
        end   = max(int((i + 1) * segment_size), start + 1)
        if augment:
            idx = random.randint(start, end - 1)
        else:
            idx = (start + end) // 2
        indices.append(min(idx, total_frames - 1))
    return indices


def read_video(path: str, num_frames: int, img_size: int, augment: bool) -> np.ndarray:
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        return np.zeros((num_frames, img_size, img_size, 3), dtype=np.float32)

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        cap.release()
        return np.zeros((num_frames, img_size, img_size, 3), dtype=np.float32)

    indices = sample_indices(total_frames, num_frames, augment)
    frames  = []

    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        success, frame = cap.read()
        if not success:
            frame = frames[-1].copy() if frames else np.zeros((img_size, img_size, 3), dtype=np.uint8)
        else:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, (img_size, img_size), interpolation=cv2.INTER_AREA)
        frames.append(frame)

    cap.release()
    return np.array(frames, dtype=np.float32) / 255.0


# NORMALIZATION

def frames_to_tensor(frames: np.ndarray) -> torch.Tensor:
    mean   = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std    = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    frames = (frames - mean) / std
    # (T, H, W, C) → (C, T, H, W)
    tensor = torch.from_numpy(frames).float().permute(3, 0, 1, 2)
    return tensor.contiguous()


# DATASET

class ASLDataset(Dataset):

    def __init__(self, excel_path, video_root, num_frames=8, img_size=160,
                 augment=False, top_n=100):
        self.video_root = Path(video_root)
        self.num_frames = num_frames
        self.img_size   = img_size
        self.augment    = augment

        df = pd.read_excel(excel_path, engine="openpyxl")

        # ── Keep only top N classes by signer diversity ──────────────────────
        vids_per_class = df.groupby("sign_label")["sign_path"].count()
        top_classes    = set(
            vids_per_class.sort_values(ascending=False).head(top_n).index.tolist()
        )
        df = df[df["sign_label"].isin(top_classes)].reset_index(drop=True)

        classes = sorted(df["sign_label"].unique().tolist())
        log.info(
            f"Using top {top_n} classes by signer diversity "
            f"(min {vids_per_class[list(top_classes)].min()} source vids per class)"
        )

        self.class2idx = {c: i for i, c in enumerate(classes)}
        self.idx2class = {i: c for c, i in self.class2idx.items()}
        self.num_classes = len(classes)
        self.samples   = []
        missing        = 0

        for _, row in df.iterrows():
            label     = row["sign_label"]
            folder    = self.video_root / row["sign_path"]
            class_idx = self.class2idx[label]
            for w in range(int(row["n_windows"])):
                p = folder / f"window_{w:03d}.mp4"
                if p.exists():
                    self.samples.append((str(p), class_idx))
                else:
                    missing += 1

        log.info(f"Loaded {len(self.samples):,} clips | {len(classes)} classes")
        if missing:
            log.warning(f"Missing files: {missing}")

    def __len__(self):
        return len(self.samples)

    def _augment(self, frames: np.ndarray) -> np.ndarray:
        """
        Augmentations safe for ASL:
          - NO horizontal flip (mirrors signs, changes meaning)
          - Random spatial crop + resize
          - Brightness / contrast jitter
          - Gaussian noise
          - Random grayscale (forces shape-based learning)
        """
        T, H, W, C = frames.shape

        # Random spatial crop (85–100%) then resize back
        if random.random() < 0.5:
            scale   = random.uniform(0.85, 1.0)
            new_H   = int(H * scale)
            new_W   = int(W * scale)
            top     = random.randint(0, H - new_H)
            left    = random.randint(0, W - new_W)
            cropped = frames[:, top:top+new_H, left:left+new_W, :]
            frames  = np.stack([
                cv2.resize(f, (W, H), interpolation=cv2.INTER_LINEAR)
                for f in cropped
            ])

        # Brightness jitter
        if random.random() < 0.4:
            frames = np.clip(frames * random.uniform(0.80, 1.20), 0, 1)

        # Contrast jitter
        if random.random() < 0.3:
            mean   = frames.mean()
            frames = np.clip((frames - mean) * random.uniform(0.8, 1.2) + mean, 0, 1)

        # Gaussian noise
        if random.random() < 0.2:
            noise  = np.random.normal(0, 0.015, frames.shape).astype(np.float32)
            frames = np.clip(frames + noise, 0, 1)

        # Random grayscale — forces model to rely on shape not color
        if random.random() < 0.15:
            gray   = frames.mean(axis=-1, keepdims=True)
            frames = np.repeat(gray, 3, axis=-1)

        return frames.astype(np.float32)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        frames      = read_video(path, self.num_frames, self.img_size, self.augment)
        if self.augment:
            frames = self._augment(frames)
        return frames_to_tensor(frames), label


# STRATIFIED SPLIT + WEIGHTED SAMPLER

def get_splits(excel_path, video_root, cfg):
    # Three separate dataset objects — augment never bleeds into val/test
    train_ds = ASLDataset(excel_path, video_root, cfg.num_frames, cfg.img_size,
                          augment=True,  top_n=cfg.top_n_classes)
    val_ds   = ASLDataset(excel_path, video_root, cfg.num_frames, cfg.img_size,
                          augment=False, top_n=cfg.top_n_classes)
    test_ds  = ASLDataset(excel_path, video_root, cfg.num_frames, cfg.img_size,
                          augment=False, top_n=cfg.top_n_classes)

    num_classes = train_ds.num_classes
    labels      = [lbl for _, lbl in train_ds.samples]
    counts      = Counter(labels)

    # Rare clips (< 5) go straight to train — shouldn't happen with top-100
    MIN_CLIPS     = 5
    rare_idx      = [i for i, l in enumerate(labels) if counts[l] < MIN_CLIPS]
    eligible_idx  = [i for i, l in enumerate(labels) if counts[l] >= MIN_CLIPS]
    eligible_lbls = [labels[i] for i in eligible_idx]

    if rare_idx:
        log.warning(f"{len(rare_idx)} clips from rare classes → train only")

    # 80 / 20 first split
    train_idx, temp_idx = train_test_split(
        eligible_idx,
        test_size    = cfg.val_split + cfg.test_split,
        stratify     = eligible_lbls,
        random_state = cfg.seed
    )
    train_idx = list(train_idx) + rare_idx

    # 20 → 50/50 val/test
    temp_lbls   = [labels[i] for i in temp_idx]
    temp_counts = Counter(temp_lbls)
    rare_temp   = [i for i in temp_idx if temp_counts[labels[i]] < 2]
    clean_temp  = [i for i in temp_idx if temp_counts[labels[i]] >= 2]
    clean_lbls  = [labels[i] for i in clean_temp]
    train_idx   = train_idx + rare_temp

    val_idx, test_idx = train_test_split(
        clean_temp,
        test_size    = 0.50,
        stratify     = clean_lbls,
        random_state = cfg.seed
    )

    log.info(
        f"Split — Train: {len(train_idx):,} | Val: {len(val_idx):,} | Test: {len(test_idx):,}"
    )

    # ── Weighted sampler: inverse class frequency ────────────────────────────
    train_labels   = [labels[i] for i in train_idx]
    train_counts   = Counter(train_labels)
    sample_weights = [1.0 / train_counts[labels[i]] for i in train_idx]
    sampler = WeightedRandomSampler(
        weights     = sample_weights,
        num_samples = len(train_idx),
        replacement = True
    )
    log.info(
        f"WeightedRandomSampler active | "
        f"imbalance ratio now ~{max(sample_weights)/min(sample_weights):.1f}x effective"
    )

    def make_train_loader(ds, idx):
        return DataLoader(
            Subset(ds, idx),
            batch_size         = cfg.batch_size,
            sampler            = sampler,
            num_workers        = cfg.num_workers,
            pin_memory         = True,
            persistent_workers = True,
            prefetch_factor    = 2,
            drop_last          = True
        )

    def make_eval_loader(ds, idx):
        return DataLoader(
            Subset(ds, idx),
            batch_size         = cfg.batch_size * 2,
            shuffle            = False,
            num_workers        = cfg.num_workers,
            pin_memory         = True,
            persistent_workers = True,
            prefetch_factor    = 2,
            drop_last          = False
        )

    return (
        make_train_loader(train_ds, train_idx),
        make_eval_loader(val_ds,   val_idx),
        make_eval_loader(test_ds,  test_idx),
        train_ds.idx2class,
        num_classes
    )


# MODEL

class ASLModel(nn.Module):

    def __init__(self, num_classes=100, dropout=0.5):
        super().__init__()
        backbone    = r3d_18(weights=R3D_18_Weights.KINETICS400_V1)
        in_features = backbone.fc.in_features
        backbone.fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_features, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout * 0.5),
            nn.Linear(512, num_classes)
        )
        self.model = backbone

    def freeze_backbone(self):
        for name, param in self.model.named_parameters():
            if "fc" not in name:
                param.requires_grad = False
        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        log.info(f"Backbone FROZEN — trainable params: {trainable:,}")

    def unfreeze_backbone(self):
        for param in self.model.parameters():
            param.requires_grad = True
        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        log.info(f"Backbone UNFROZEN — trainable params: {trainable:,}")

    def forward(self, x):
        return self.model(x)


# METRICS

@torch.no_grad()
def top1_accuracy(logits, labels):
    return (logits.argmax(dim=1) == labels).float().mean().item() * 100


@torch.no_grad()
def topk_accuracy(logits, labels, k=5):
    _, top_k  = logits.topk(k, dim=1)
    correct   = top_k.eq(labels.unsqueeze(1).expand_as(top_k))
    return correct.any(dim=1).float().mean().item() * 100


# TRAIN
def train_epoch(model, loader, optimizer, criterion, scaler, device, epoch, grad_clip):
    model.train()
    total_loss = total_top1 = total_top5 = 0
    pbar = tqdm(loader, desc=f"Epoch {epoch:03d} [train]")

    for videos, labels in pbar:
        videos = videos.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)

        with torch.autocast(device_type="cuda"):
            outputs = model(videos)
            loss    = criterion(outputs, labels)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()
        total_top1 += top1_accuracy(outputs, labels)
        total_top5 += topk_accuracy(outputs, labels, k=5)
        n = pbar.n + 1
        pbar.set_postfix(
            loss=f"{total_loss/n:.4f}",
            top1=f"{total_top1/n:.1f}%",
            top5=f"{total_top5/n:.1f}%"
        )

    n = len(loader)
    return total_loss / n, total_top1 / n, total_top5 / n


# EVALUATE

@torch.no_grad()
def evaluate(model, loader, criterion, device, epoch, split="val"):
    model.eval()
    total_loss = total_top1 = total_top5 = 0
    pbar = tqdm(loader, desc=f"Epoch {epoch:03d} [{split}]")

    for videos, labels in pbar:
        videos = videos.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        with torch.autocast(device_type="cuda"):
            outputs = model(videos)
            loss    = criterion(outputs, labels)

        total_loss += loss.item()
        total_top1 += top1_accuracy(outputs, labels)
        total_top5 += topk_accuracy(outputs, labels, k=5)
        n = pbar.n + 1
        pbar.set_postfix(
            loss=f"{total_loss/n:.4f}",
            top1=f"{total_top1/n:.1f}%",
            top5=f"{total_top5/n:.1f}%"
        )

    n = len(loader)
    return total_loss / n, total_top1 / n, total_top5 / n


# PER-CLASS ACCURACY


@torch.no_grad()
def per_class_accuracy(model, loader, device, idx2class):
    model.eval()
    correct = Counter()
    total   = Counter()

    for videos, labels in tqdm(loader, desc="Per-class eval"):
        videos = videos.to(device, non_blocking=True)
        preds  = model(videos).argmax(dim=1).cpu()
        labels = labels.cpu()
        for p, l in zip(preds, labels):
            total[l.item()]   += 1
            correct[l.item()] += int(p == l)

    results = []
    for cls_idx in sorted(total.keys()):
        acc = 100.0 * correct[cls_idx] / total[cls_idx] if total[cls_idx] > 0 else 0.0
        results.append((idx2class[cls_idx], acc, correct[cls_idx], total[cls_idx]))

    results.sort(key=lambda x: x[1])

    log.info("\n=== PER-CLASS ACCURACY (worst → best) ===")
    log.info(f"{'Class':<25} {'Acc':>7}  {'Correct':>7} / {'Total':>5}")
    log.info("-" * 55)
    for name, acc, corr, tot in results[:20]:
        log.info(f"{name:<25} {acc:>6.1f}%  {corr:>7} / {tot:>5}   ← worst 20")
    log.info("  ...")
    for name, acc, corr, tot in results[-20:]:
        log.info(f"{name:<25} {acc:>6.1f}%  {corr:>7} / {tot:>5}   ← best 20")

    mean_per_class = sum(r[1] for r in results) / len(results)
    log.info(f"\nMean per-class accuracy: {mean_per_class:.2f}%")
    return results


# MAIN

def main():
    cfg = get_config()
    seed_everything(cfg.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"Device: {device}")
    if device.type == "cuda":
        log.info(f"GPU: {torch.cuda.get_device_name(0)}")

    train_loader, val_loader, test_loader, idx2class, num_classes = get_splits(
        cfg.excel, cfg.video_root, cfg
    )
    log.info(f"Training on {num_classes} classes")

    model     = ASLModel(num_classes=num_classes, dropout=cfg.dropout).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=cfg.label_smoothing)
    scaler    = torch.cuda.amp.GradScaler()

    os.makedirs(cfg.ckpt_dir, exist_ok=True)

    # ── Phase 1: backbone frozen ─────────────────────────────────────────────
    model.freeze_backbone()
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=cfg.lr_frozen, weight_decay=cfg.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg.freeze_epochs, eta_min=cfg.lr_frozen * 0.1
    )

    best_val_top1 = 0.0
    best_val_top5 = 0.0

    for epoch in range(1, cfg.epochs + 1):
        t0 = time.time()

        # ── Switch to Phase 2: unfreeze backbone ─────────────────────────────
        if epoch == cfg.freeze_epochs + 1:
            model.unfreeze_backbone()
            optimizer = optim.AdamW(
                model.parameters(),
                lr=cfg.lr_unfrozen, weight_decay=cfg.weight_decay
            )
            remaining = cfg.epochs - cfg.freeze_epochs
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=remaining, eta_min=1e-6
            )
            log.info(f"  [Phase 2] Full fine-tuning for {remaining} epochs")

        tr_loss, tr_top1, tr_top5 = train_epoch(
            model, train_loader, optimizer, criterion, scaler, device, epoch, cfg.grad_clip
        )
        vl_loss, vl_top1, vl_top5 = evaluate(
            model, val_loader, criterion, device, epoch, "val"
        )
        scheduler.step()

        elapsed = (time.time() - t0) / 60
        phase   = "frozen" if epoch <= cfg.freeze_epochs else "full"
        log.info(
            f"\nEpoch {epoch}/{cfg.epochs} [{phase}] | "
            f"LR: {scheduler.get_last_lr()[0]:.2e} | {elapsed:.1f}min"
        )
        log.info(f"  Train — Loss: {tr_loss:.4f} | Top1: {tr_top1:.2f}% | Top5: {tr_top5:.2f}%")
        log.info(f"  Val   — Loss: {vl_loss:.4f} | Top1: {vl_top1:.2f}% | Top5: {vl_top5:.2f}%")

        # Save last checkpoint with current epoch's real val acc
        torch.save(
            {
                "model":     model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "val_top1":  vl_top1,
                "val_top5":  vl_top5,
                "epoch":     epoch,
            },
            os.path.join(cfg.ckpt_dir, "last_checkpoint.pt")
        )

        if vl_top1 > best_val_top1:
            best_val_top1 = vl_top1
            best_val_top5 = vl_top5
            torch.save(
                {
                    "model":       model.state_dict(),
                    "optimizer":   optimizer.state_dict(),
                    "scheduler":   scheduler.state_dict(),
                    "val_top1":    best_val_top1,
                    "val_top5":    best_val_top5,
                    "epoch":       epoch,
                    "idx2class":   idx2class,
                    "num_classes": num_classes,
                },
                os.path.join(cfg.ckpt_dir, "best_model.pt")
            )
            log.info(
                f"  ★ BEST MODEL SAVED — "
                f"Top1: {best_val_top1:.2f}% | Top5: {best_val_top5:.2f}%"
            )

    # ── Final test evaluation ────────────────────────────────────────────────
    log.info("\nLoading best model for final test evaluation...")
    ckpt = torch.load(os.path.join(cfg.ckpt_dir, "best_model.pt"), map_location=device)
    model.load_state_dict(ckpt["model"])

    te_loss, te_top1, te_top5 = evaluate(model, test_loader, criterion, device, 0, "test")

    log.info(f"\n{'='*55}")
    log.info(f"  FINAL TEST  — Top1: {te_top1:.2f}% | Top5: {te_top5:.2f}%")
    log.info(f"  BEST  VAL   — Top1: {best_val_top1:.2f}% | Top5: {best_val_top5:.2f}%")
    log.info(f"{'='*55}")

    per_class_accuracy(model, test_loader, device, idx2class)


if __name__ == "__main__":
    main()