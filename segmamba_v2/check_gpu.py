"""Check the CUDA build (causal-conv1d, mamba) and peak GPU memory before the long run (about a minute).

Run:  python segmamba_v2/check_gpu.py
      python segmamba_v2/check_gpu.py --batch_size 1
Builds SegMamba with the configured channels, runs 3 training steps at the configured batch and patch size
(AMP + deep-supervision loss, as in train.py), then one sliding-window inference over a 162x256x256 volume
(the largest MS3SEG FLAIR at the network's spacing, before cropping to the brain).
"""
import config as cfg  # must be first

import argparse

import torch
from monai.inferers import SlidingWindowInferer

from losses import DeepSupervisionLoss, DiceBCELoss

GB = 1024 ** 3
INFER_SHAPE = (162, 256, 256)


def train_peak(batch_size):
    model = cfg.build_model().to(cfg.device).train()
    optimizer = torch.optim.SGD(model.parameters(), lr=cfg.sgd_lr, momentum=0.99, nesterov=True)
    scaler = torch.cuda.amp.GradScaler()
    loss_fn = DeepSupervisionLoss(DiceBCELoss(batch_dice=cfg.batch_dice))
    x = torch.randn(batch_size, len(cfg.modalities), *cfg.roi_size, device=cfg.device)
    y = (torch.rand(batch_size, cfg.num_classes, *cfg.roi_size, device=cfg.device) > 0.99).float()

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
    x = torch.randn(1, len(cfg.modalities), *INFER_SHAPE, device=cfg.device)
    torch.cuda.reset_peak_memory_stats()
    with torch.no_grad(), torch.autocast("cuda"):
        out = infer(x, model)
    torch.cuda.synchronize()
    peak = torch.cuda.max_memory_reserved() / GB
    assert out.shape == (1, cfg.num_classes, *INFER_SHAPE), out.shape
    del model, x, out
    torch.cuda.empty_cache()
    return peak


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=cfg.batch_size)
    args, _ = parser.parse_known_args()

    if not torch.cuda.is_available():
        raise SystemExit("no CUDA GPU visible to PyTorch; SegMamba's Mamba layers need one even for inference")
    total = torch.cuda.get_device_properties(cfg.device).total_memory / GB
    print(f"GPU: {torch.cuda.get_device_name(cfg.device)} ({total:.1f} GB)")
    print(f"in_chans={len(cfg.modalities)} ({'+'.join(cfg.modalities)}) out_chans={cfg.num_classes} roi={cfg.roi_size}")

    try:
        peak = train_peak(args.batch_size)
        print(f"training   batch_size={args.batch_size}: peak {peak:.2f} GB ({peak / total:.0%} of GPU)")
        if peak > 0.9 * total:
            print("  -> close to the limit: set batch_size = 1 in config.py")
    except torch.cuda.OutOfMemoryError:
        print(f"training   batch_size={args.batch_size}: OUT OF MEMORY (needs > {total:.1f} GB) -> set batch_size = 1 in config.py")
        torch.cuda.empty_cache()

    try:
        print(f"inference  sw_batch_size={cfg.sw_batch_size}: peak {inference_peak():.2f} GB")
    except torch.cuda.OutOfMemoryError:
        print(f"inference  sw_batch_size={cfg.sw_batch_size}: OUT OF MEMORY -> set sw_batch_size = 1 in config.py")
    print("CUDA build OK")
