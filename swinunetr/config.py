"""Shared settings for the Swin UNETR baseline.

Everything that affects the comparison with SegMamba (data, split, patch size,
epochs, label mapping) is kept identical to 3_train.py / 4_predict.py.
Only the network and its optimizer differ.
"""
import os
import sys
import inspect

# Scripts live in <repo>/swinunetr/ but use the repo's light_training and the
# vendored monai (same versions SegMamba uses), with repo-relative paths.
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
os.chdir(ROOT)

# Reduce fragmentation of the CUDA caching allocator (helps avoid OOM on 24 GB).
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch

# ---------------- data (same as SegMamba) ----------------
# Anchor paths at the repository root so running from any CWD is safe.
data_dir = os.path.join(ROOT, "data", "fullres", "train")
raw_data_dir = os.path.join(ROOT, "data", "BraTS2020_TrainingData", "MICCAI_BraTS2020_TrainingData")
roi_size = [128, 128, 128]

# ---------------- training (same budget as SegMamba) ----------------
logdir = "logs/swinunetr"
model_save_path = os.path.join(logdir, "model")
split_file = os.path.join(logdir, "data_split.json")
env = "pytorch"
max_epoch = 250
batch_size = 2
val_every = 2
num_gpus = 1
device = "cuda:0"
augmentation = True

# ---------------- Swin UNETR ----------------
# feature_size=48 is the configuration used for BraTS in the Swin UNETR paper.
# use_checkpoint=True (gradient checkpointing) is what keeps batch 2 @ 128^3
# well inside 24 GB; run check_gpu_memory.py to see the actual peak.
feature_size = 48
use_checkpoint = True
drop_rate = 0.0
attn_drop_rate = 0.0
dropout_path_rate = 0.0

# Transformers train poorly with SGD lr=1e-2/momentum 0.99 (SegMamba's setup),
# so we use the Swin UNETR recipe: AdamW + warmup + cosine decay.
# Set optimizer_name = "sgd" to reproduce SegMamba's optimizer exactly.
optimizer_name = "adamw"
lr = 1e-4
weight_decay = 1e-5
warmup_ratio = 0.03  # fraction of total steps

# "ce" = same loss as SegMamba (4-class cross entropy).
# "dice_ce" = CE + soft Dice, usually better for the small ET region.
loss_name = "ce"

# ---------------- inference (same as SegMamba 4_predict.py) ----------------
save_path = "prediction_results/swinunetr"
sw_batch_size = 2
sw_overlap = 0.5
mirror_axes = [0, 1, 2]


def build_model():
    from monai.networks.nets import SwinUNETR

    kwargs = dict(
        in_channels=4,
        out_channels=4,  # background, NCR/NET (1), ED (2), ET (4 -> 3)
        feature_size=feature_size,
        drop_rate=drop_rate,
        attn_drop_rate=attn_drop_rate,
        dropout_path_rate=dropout_path_rate,
        use_checkpoint=use_checkpoint,
    )

    # MONAI <= 1.4 may still require img_size; newer versions deprecate it.
    sig = inspect.signature(SwinUNETR.__init__)
    img_size_param = sig.parameters.get("img_size")
    if img_size_param is not None and img_size_param.default is inspect._empty:
        kwargs["img_size"] = roi_size

    return SwinUNETR(**kwargs)


def map_brats_labels(label):
    """BraTS2020 labels {0,1,2,4} -> {0,1,2,3}; padding (-1) -> 0. Same as 3_train.py."""
    label = torch.where(label == 4, torch.full_like(label, 3), label)
    label = torch.where(label < 0, torch.zeros_like(label), label)
    return label


def convert_labels(labels, dim=1):
    """TC, WT, ET region masks, same order as SegMamba."""
    result = [(labels == 1) | (labels == 3), (labels == 1) | (labels == 3) | (labels == 2), labels == 3]
    return torch.cat(result, dim=dim).float()


def split_names(ds):
    return sorted(os.path.basename(p)[:-4] for p in ds.datalist)
