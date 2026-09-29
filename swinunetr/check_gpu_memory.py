"""Measure peak GPU memory of Swin UNETR training and inference before the long run.

Run:  python swinunetr/check_gpu_memory.py   # With gradient checkpointing (default)
      python swinunetr/check_gpu_memory.py --no_checkpoint   # Without gradient checkpointing

"""
import config as cfg  # must be first

import argparse

import torch
import torch.nn as nn
from monai.inferers import SlidingWindowInferer

GB = 1024 ** 3


def train_peak(batch_size):
    model = cfg.build_model().to(cfg.device).train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr)
    scaler = torch.cuda.amp.GradScaler()
    loss_fn = nn.CrossEntropyLoss()
    x = torch.randn(batch_size, 4, *cfg.roi_size, device=cfg.device)
    y = torch.randint(0, 4, (batch_size, *cfg.roi_size), device=cfg.device)

    torch.cuda.reset_peak_memory_stats()
    for _ in range(3):
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda"):
            loss = loss_fn(model(x), y)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
    torch.cuda.synchronize()
    peak = torch.cuda.max_memory_reserved() / GB
    del model, optimizer, x, y
    torch.cuda.empty_cache()
    return peak


def inference_peak():
    model = cfg.build_model().to(cfg.device).eval()
    infer = SlidingWindowInferer(roi_size=cfg.roi_size, sw_batch_size=cfg.sw_batch_size,
                                 overlap=cfg.sw_overlap, mode="gaussian")
    x = torch.randn(1, 4, 155, 240, 240, device=cfg.device)  # full, uncropped BraTS volume (worst case)
    torch.cuda.reset_peak_memory_stats()
    with torch.no_grad(), torch.autocast("cuda"):
        infer(x, model)
    torch.cuda.synchronize()
    peak = torch.cuda.max_memory_reserved() / GB
    del model, x
    torch.cuda.empty_cache()
    return peak


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--no_checkpoint", action="store_true", help="disable gradient checkpointing")
    parser.add_argument("--batch_size", type=int, default=cfg.batch_size)
    args = parser.parse_args()
    if args.no_checkpoint:
        cfg.use_checkpoint = False
    cfg.batch_size = args.batch_size

    total = torch.cuda.get_device_properties(cfg.device).total_memory / GB
    print(f"GPU: {torch.cuda.get_device_name(cfg.device)} ({total:.1f} GB)")
    print(f"feature_size={cfg.feature_size} use_checkpoint={cfg.use_checkpoint} roi={cfg.roi_size}")

    try:
        peak = train_peak(cfg.batch_size)
        print(f"training   batch_size={cfg.batch_size}: peak {peak:.2f} GB ({peak / total:.0%} of GPU)")
        if peak > 0.9 * total:
            print("  -> close to the limit: set use_checkpoint=True or batch_size=1 in config.py")
    except torch.cuda.OutOfMemoryError:
        print(f"training   batch_size={cfg.batch_size}: OUT OF MEMORY (needs > {total:.1f} GB) -> set use_checkpoint=True or batch_size=1 in config.py")
        torch.cuda.empty_cache()

    try:
        print(f"inference  sw_batch_size={cfg.sw_batch_size}: peak {inference_peak():.2f} GB")
    except torch.cuda.OutOfMemoryError:
        print(f"inference  sw_batch_size={cfg.sw_batch_size}: OUT OF MEMORY -> set sw_batch_size=1 in config.py")
