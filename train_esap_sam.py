#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Standalone binary facility segmentation training. Paper recipe, not reference-script tricks."""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import random
import sys
import time

import numpy as np
from PIL import Image
import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from esap_sam import ESAPSAM


def environment_report(strict: bool = False) -> dict:
    try:
        triton_version = importlib.metadata.version("triton")
    except importlib.metadata.PackageNotFoundError:
        triton_version = None
    report = {
        "python": platform.python_version(), "platform": platform.platform(),
        "torch": torch.__version__, "cuda_runtime": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(), "triton": triton_version,
        "numpy": np.__version__, "Pillow": importlib.metadata.version("Pillow"),
        "tqdm": importlib.metadata.version("tqdm"),
        "cuda_available": torch.cuda.is_available(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }
    expected = {"python": "3.13.12", "torch": "2.10.0+cu129",
                "cuda_runtime": "12.9", "cudnn": 91002, "triton": "3.6.0"}
    report["target_mismatches"] = {k: {"actual": report[k], "expected": v}
                                     for k, v in expected.items() if report[k] != v}
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if strict and report["target_mismatches"]:
        raise RuntimeError("Environment differs from the supplied screenshot; see target_mismatches")
    return report


def seed_all(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def seed_worker(worker_id: int):
    seed = torch.initial_seed() % 2**32
    random.seed(seed)
    np.random.seed(seed)


def index_files(folder: Path, extensions: set[str]) -> dict[str, Path]:
    if not folder.is_dir():
        raise FileNotFoundError(f"Missing data directory: {folder}")
    result = {}
    for path in sorted(folder.rglob("*")):
        if path.is_file() and path.suffix.lower() in extensions:
            key = path.relative_to(folder).with_suffix("").as_posix()
            if key in result:
                raise ValueError(f"Duplicate relative stem {key}: {result[key]} and {path}")
            result[key] = path
    return result


def pair_files(image_dir: Path, mask_dir: Path) -> list[tuple[Path, Path]]:
    images = index_files(image_dir, {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"})
    masks = index_files(mask_dir, {".png", ".tif", ".tiff", ".bmp"})
    if not images:
        raise ValueError(f"No images in {image_dir}")
    if images.keys() != masks.keys():
        raise ValueError(f"Image/mask relative stems must match. Missing masks: "
                         f"{sorted(images.keys() - masks.keys())[:10]}; extra masks: "
                         f"{sorted(masks.keys() - images.keys())[:10]}")
    return [(images[key], masks[key]) for key in sorted(images)]


def make_manifest(root: Path, seed: int, ratios: list[float]) -> dict:
    splits = {}
    if (root / "train").is_dir():
        for name in ("train", "val", "test"):
            if name == "test" and not (root / name).exists():
                continue
            splits[name] = pair_files(root / name / "images", root / name / "masks")
    else:
        pairs = pair_files(root / "images", root / "masks")
        if len(pairs) < 3:
            raise ValueError("Automatic train/val/test splitting requires at least 3 image-mask pairs")
        random.Random(seed).shuffle(pairs)
        n = len(pairs)
        n_test = max(1, int(n * ratios[2]))
        n_val = max(1, int(n * ratios[1]))
        n_train = n - n_val - n_test
        if n_train < 1:
            raise ValueError("Split ratios leave an empty training set")
        splits = {"train": pairs[:n_train], "val": pairs[n_train:n_train + n_val],
                  "test": pairs[n_train + n_val:]}
    return {name: [[im.relative_to(root).as_posix(), mask.relative_to(root).as_posix()]
                   for im, mask in pairs] for name, pairs in splits.items()}


def validate_manifest(root: Path, manifest: dict):
    if not manifest.get("train") or not manifest.get("val"):
        raise ValueError("Both train and val sets must be nonempty")
    seen = set()
    for split, pairs in manifest.items():
        for image, mask in pairs:
            for relative in (image, mask):
                resolved = (root / relative).resolve()
                if not resolved.is_relative_to(root) or not resolved.is_file():
                    raise ValueError(f"Missing file or path outside data root: {relative}")
            image_path = (root / image).resolve()
            if image_path in seen:
                raise ValueError(f"Image reused across splits: {image_path}")
            seen.add(image_path)


class FacilityDataset(Dataset):
    """RGB input, scalar/identical-channel binary or indexed masks. No augmentation by default."""
    def __init__(self, root: Path, pairs: list, image_size=None,
                 foreground_values=None, ignore_index=None):
        self.root, self.pairs, self.image_size = root, pairs, image_size
        self.foreground_values, self.ignore_index = foreground_values, ignore_index

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, index):
        image_name, mask_name = self.pairs[index]
        with Image.open(self.root / image_name) as source:
            image = source.convert("RGB")
        with Image.open(self.root / mask_name) as source:
            # Preserve palette/index IDs, not luminance-converted palette colors.
            mask_array = np.array(source)
        if mask_array.ndim == 3:
            if mask_array.shape[-1] not in (3, 4) or not np.all(mask_array[..., :3] == mask_array[..., :1]):
                raise ValueError(f"Color masks need explicit conversion to scalar class IDs: {mask_name}")
            mask_array = mask_array[..., 0]
        if mask_array.ndim != 2 or mask_array.shape != (image.height, image.width):
            raise ValueError(f"Image/mask size or mask dimensionality mismatch: {image_name}, {mask_name}")
        ignored = mask_array == self.ignore_index if self.ignore_index is not None else np.zeros_like(mask_array, bool)
        values = set(np.unique(mask_array[~ignored]).tolist())
        if self.foreground_values is None:
            if not values.issubset({0, 1, 255}):
                raise ValueError(f"Mask {mask_name} has IDs {values}; specify --foreground-values for facility IDs")
            foreground = mask_array > 0
        else:
            allowed = {0, *self.foreground_values}
            if not values.issubset(allowed):
                raise ValueError(f"Mask {mask_name} has undeclared class IDs {values - allowed}")
            foreground = np.isin(mask_array, self.foreground_values)
        target = foreground.astype(np.float32)
        target[ignored] = -1
        if self.image_size is not None:
            height, width = self.image_size
            image = image.resize((width, height), Image.Resampling.BILINEAR)
            mask_image = Image.fromarray(target)
            target = np.array(mask_image.resize((width, height), Image.Resampling.NEAREST))
        pixels = np.array(image, dtype=np.float32) / 255.0
        return {"image": torch.from_numpy(pixels.transpose(2, 0, 1).copy()),
                "mask": torch.from_numpy(target.copy()).unsqueeze(0), "name": image_name}


def collate_batch(samples: list[dict]) -> dict:
    # Variable image sizes are padded within a batch; -1 pixels never enter loss/metrics.
    height = max(item["image"].shape[-2] for item in samples)
    width = max(item["image"].shape[-1] for item in samples)
    images, masks = [], []
    for item in samples:
        h, w = item["image"].shape[-2:]
        images.append(F.pad(item["image"], (0, width - w, 0, height - h)))
        masks.append(F.pad(item["mask"], (0, width - w, 0, height - h), value=-1))
    return {"image": torch.stack(images), "mask": torch.stack(masks),
            "name": [item["name"] for item in samples]}


class DiceLoss(nn.Module):
    """Foreground soft Dice per image; average over images with at least one valid pixel."""
    def __init__(self, smooth: float = 1e-5):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits: Tensor, target: Tensor) -> Tensor:
        valid = target >= 0
        active = valid.flatten(1).any(1)
        if not active.any():
            raise ValueError("Batch contains no labeled pixels")
        probabilities = logits.float().sigmoid() * valid
        truth = target.float().clamp_min(0) * valid
        intersection = (probabilities * truth).flatten(1).sum(1)
        denominator = probabilities.flatten(1).sum(1) + truth.flatten(1).sum(1)
        losses = 1 - (2 * intersection + self.smooth) / (denominator + self.smooth)
        return losses[active].mean()


class BinaryMetrics:
    """Accumulate the whole split's confusion matrix, independent of batch partitioning."""
    def __init__(self, device, threshold=0.5):
        self.counts = torch.zeros(4, dtype=torch.int64, device=device)
        self.threshold = threshold

    @torch.no_grad()
    def update(self, logits: Tensor, target: Tensor):
        valid = target >= 0
        prediction = logits.float().sigmoid() > self.threshold
        truth = target > 0.5
        self.counts += torch.stack(((prediction & truth & valid).sum(),
                                    (prediction & ~truth & valid).sum(),
                                    (~prediction & truth & valid).sum(),
                                    (~prediction & ~truth & valid).sum()))

    def compute(self) -> dict:
        tp, fp, fn, tn = self.counts.cpu().tolist()
        def divide(num, den):
            return num / den if den else 0.0
        iou_fg = divide(tp, tp + fp + fn)
        iou_bg = divide(tn, tn + fp + fn)
        present_ious = [value for value, den in ((iou_fg, tp + fp + fn), (iou_bg, tn + fp + fn)) if den]
        dice = divide(2 * tp, 2 * tp + fp + fn)
        return {"Dice": dice, "IoU": iou_fg, "mIoU": sum(present_ious) / len(present_ious) if present_ious else 0.0,
                "Acc": divide(tp + tn, tp + fp + fn + tn), "F1": dice,
                "Pre": divide(tp, tp + fp), "Rec": divide(tp, tp + fn),
                "TP": tp, "FP": fp, "FN": fn, "TN": tn}


def run_epoch(loader, model, criterion, device, optimizer=None, precision="fp32", desc=""):
    training = optimizer is not None
    model.train(training)
    confusion = BinaryMetrics(device, loader.threshold)
    total_loss, num_images = 0., 0
    for batch in tqdm(loader, desc=desc, leave=False):
        images = batch["image"].to(device, non_blocking=True)
        target = batch["mask"].to(device, non_blocking=True)
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=precision == "bf16"):
                logits = model(images)
            # FP32 loss outside autocast; no BCE, EMA, auxiliary supervision, or clipping.
            loss = criterion(logits, target)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Nonfinite loss on {batch['name']}")
            if training:
                loss.backward()
                if not all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None):
                    raise FloatingPointError(f"Nonfinite gradients on {batch['name']}")
                optimizer.step()
        active_images = int((target >= 0).flatten(1).any(1).sum())
        total_loss += loss.detach().item() * active_images
        num_images += active_images
        confusion.update(logits.detach(), target)
    if not num_images:
        raise ValueError("Empty or entirely unlabeled data loader")
    return {"Loss": total_loss / num_images, **confusion.compute()}


def save_atomic(path: Path, payload: dict):
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def capture_rng(generator: torch.Generator) -> dict:
    numpy_state = np.random.get_state()
    return {"python": random.getstate(), "numpy": [numpy_state[0], numpy_state[1].tolist(),
                                                   *numpy_state[2:]],
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
            "loader": generator.get_state()}


def restore_rng(state: dict, generator: torch.Generator):
    random.setstate(state["python"])
    ns = state["numpy"]
    np.random.set_state((ns[0], np.asarray(ns[1], dtype=np.uint32), ns[2], ns[3], ns[4]))
    torch.set_rng_state(state["torch"].cpu())
    if torch.cuda.is_available() and state["cuda"]:
        if len(state["cuda"]) != torch.cuda.device_count():
            raise ValueError("Resume GPU count differs from checkpoint RNG state")
        torch.cuda.set_rng_state_all([value.cpu() for value in state["cuda"]])
    generator.set_state(state["loader"].cpu())


def get_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--data-dir", "--data_dir", type=Path)
    p.add_argument("--sam-checkpoint", type=Path)
    p.add_argument("--backbone", choices=("vit_l", "vit_b"), default="vit_l")
    p.add_argument("--adapter-channels", type=int, default=32)
    p.add_argument("--directional-kernel", type=int, default=7)
    p.add_argument("--hp-hidden", type=int, default=512)
    p.add_argument("--epochs", type=int, default=300)
    p.add_argument("--batch-size", "--batch_size", type=int, default=2)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=1e-5)
    p.add_argument("--betas", nargs=2, type=float, default=[0.9, 0.999])
    p.add_argument("--num-workers", "--num_workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--image-size", nargs=2, type=int, metavar=("HEIGHT", "WIDTH"),
                   help="Optional resize; default preserves prepared image/patch dimensions")
    p.add_argument("--foreground-values", nargs="+", type=int,
                   help="Facility class IDs to merge, e.g. 1 2 3. Default accepts binary 0/1 or 0/255 masks")
    p.add_argument("--ignore-index", type=int)
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--split-ratios", nargs=3, type=float, default=[0.7, 0.2, 0.1])
    p.add_argument("--precision", choices=("fp32", "bf16"), default="fp32")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--output-dir", type=Path, help="New run directory (must not already contain files)")
    p.add_argument("--resume", type=Path, help="Full checkpoint; restores model, optimizer, splits and RNG")
    p.add_argument("--eval-only", action="store_true", help="Requires --resume; evaluates without training")
    p.add_argument("--eval-split", choices=("val", "test"), default="test")
    p.add_argument("--check-env", action="store_true")
    p.add_argument("--strict-env", action="store_true")
    p.add_argument("--allow-random-init", action="store_true", help="SMOKE TEST ONLY: bypass pretrained weights")
    args = p.parse_args()
    return args


def main():
    args = get_args()
    environment = environment_report(args.strict_env)
    if args.check_env:
        return
    if args.data_dir is None:
        raise ValueError("--data-dir is required")
    if args.eval_only and args.resume is None:
        raise ValueError("--eval-only requires --resume")
    checkpoint = torch.load(args.resume, map_location="cpu", weights_only=True) if args.resume else None
    if checkpoint is not None:
        # Preserve the original experiment; runtime paths/device/workers and target epochs may change.
        for name in ("batch_size", "lr", "weight_decay", "betas", "seed", "image_size",
                     "foreground_values", "ignore_index", "threshold", "split_ratios"):
            setattr(args, name, checkpoint["train_config"][name])
        for name, value in checkpoint["model_config"].items():
            setattr(args, name, value)
    if args.epochs < 1 or args.batch_size < 1 or args.num_workers < 0:
        raise ValueError("epochs/batch-size must be positive; num-workers must be nonnegative")
    if args.lr <= 0 or args.weight_decay < 0 or not all(0 <= beta < 1 for beta in args.betas):
        raise ValueError("Invalid AdamW hyperparameters")
    if not 0 < args.threshold < 1 or any(r <= 0 for r in args.split_ratios) or abs(sum(args.split_ratios) - 1) > 1e-6:
        raise ValueError("threshold must be in (0,1); positive split ratios must sum to 1")
    if args.image_size is not None and min(args.image_size) < 16:
        raise ValueError("--image-size dimensions must be >= 16")
    if args.foreground_values is not None and (0 in args.foreground_values or args.ignore_index in args.foreground_values):
        raise ValueError("Foreground IDs must exclude background 0 and the ignore index")
    device = torch.device(args.device)
    if device.type not in {"cuda", "cpu"}:
        raise ValueError("Supported devices: cpu and cuda[:index]")
    if args.precision == "bf16" and (device.type != "cuda" or not torch.cuda.is_bf16_supported()):
        raise ValueError("--precision bf16 requires a CUDA device with BF16 support; use fp32")
    seed_all(args.seed)
    root = args.data_dir.resolve()
    manifest = checkpoint["data_manifest"] if checkpoint else make_manifest(root, args.seed, args.split_ratios)
    validate_manifest(root, manifest)
    generator = torch.Generator().manual_seed(args.seed)
    loaders = {}
    for split, pairs in manifest.items():
        dataset = FacilityDataset(root, pairs, args.image_size, args.foreground_values, args.ignore_index)
        loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=split == "train",
                            num_workers=args.num_workers, pin_memory=device.type == "cuda",
                            worker_init_fn=seed_worker, generator=generator if split == "train" else None,
                            collate_fn=collate_batch, persistent_workers=False)
        loader.threshold = args.threshold
        loaders[split] = loader
    print("Dataset sizes:", {key: len(value.dataset) for key, value in loaders.items()})
    model = ESAPSAM(backbone=args.backbone,
                    sam_checkpoint=str(args.sam_checkpoint) if args.sam_checkpoint and not checkpoint else None,
                    adapter_channels=args.adapter_channels, directional_kernel=args.directional_kernel,
                    hp_hidden=args.hp_hidden, allow_random_init=bool(checkpoint) or args.allow_random_init)
    if checkpoint:
        model.load_state_dict(checkpoint["model"], strict=True)
    model.to(device)
    print(f"Parameters: total={sum(p.numel() for p in model.parameters()):,}; "
          f"trainable={sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    criterion = DiceLoss()
    if args.eval_only:
        if args.eval_split not in loaders:
            raise ValueError(f"No {args.eval_split} split in checkpoint manifest")
        metrics = run_epoch(loaders[args.eval_split], model, criterion, device,
                            precision=args.precision, desc=args.eval_split)
        print(json.dumps({"split": args.eval_split, **metrics}, indent=2))
        if args.output_dir:
            args.output_dir.mkdir(parents=True, exist_ok=True)
            path = args.output_dir / f"{args.eval_split}_metrics.json"
            if path.exists():
                raise FileExistsError(path)
            path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        return
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                                  lr=args.lr, weight_decay=args.weight_decay, betas=tuple(args.betas))
    start_epoch, best_miou = 1, -1.0
    if checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer"])
        start_epoch, best_miou = checkpoint["epoch"] + 1, checkpoint["best_val_miou"]
        restore_rng(checkpoint["rng"], generator)
    if start_epoch > args.epochs:
        raise ValueError(f"Checkpoint already reached epoch {start_epoch - 1}; increase --epochs")
    output = args.output_dir or Path("runs") / f"ESAP_SAM_{args.backbone}_{dt.datetime.now():%Y%m%d_%H%M%S_%f}"
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Output directory is nonempty: {output}; choose a new --output-dir")
    output.mkdir(parents=True, exist_ok=True)
    (output / "weights").mkdir()
    config = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    (output / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    (output / "environment.json").write_text(json.dumps(environment, indent=2), encoding="utf-8")
    (output / "splits.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    random_init = checkpoint["random_backbone"] if checkpoint else args.sam_checkpoint is None
    if random_init:
        print("WARNING: RANDOM FROZEN BACKBONE. This run is a pipeline smoke test, not a paper experiment.")
    print("Training: Dice only; constant LR; AdamW; no EMA, scheduler, warmup, clipping or augmentation.")
    metric_names = ["Loss", "Dice", "IoU", "mIoU", "Acc", "F1", "Pre", "Rec", "TP", "FP", "FN", "TN"]
    with (output / "history.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["epoch", "seconds", "lr"] +
                                [f"{split}_{name}" for split in ("train", "val") for name in metric_names])
        writer.writeheader()
        for epoch in range(start_epoch, args.epochs + 1):
            started = time.perf_counter()
            train = run_epoch(loaders["train"], model, criterion, device, optimizer,
                              args.precision, f"Epoch {epoch}/{args.epochs} train")
            val = run_epoch(loaders["val"], model, criterion, device,
                            precision=args.precision, desc=f"Epoch {epoch}/{args.epochs} val")
            row = {"epoch": epoch, "seconds": time.perf_counter() - started,
                   "lr": optimizer.param_groups[0]["lr"],
                   **{f"train_{k}": v for k, v in train.items()}, **{f"val_{k}": v for k, v in val.items()}}
            writer.writerow(row)
            stream.flush()
            improved = val["mIoU"] > best_miou
            best_miou = max(best_miou, val["mIoU"])
            payload = {"format_version": 1, "epoch": epoch, "model_config": model.config,
                       "train_config": config, "model": model.state_dict(), "optimizer": optimizer.state_dict(),
                       "best_val_miou": best_miou, "train_metrics": train, "val_metrics": val,
                       "data_manifest": manifest, "rng": capture_rng(generator), "random_backbone": random_init}
            save_atomic(output / "weights" / "last.pt", payload)
            if improved:
                save_atomic(output / "weights" / "best.pt", payload)
            print(f"Epoch {epoch}/{args.epochs} | train DiceLoss={train['Loss']:.5f} "
                  f"| val DiceLoss={val['Loss']:.5f} F1={val['F1']:.5f} mIoU={val['mIoU']:.5f} "
                  f"| best={best_miou:.5f} | {row['seconds']:.1f}s")
    print(f"Finished. Best validation mIoU={best_miou:.6f}. Results: {output.resolve()}")


if __name__ == "__main__":
    main()
