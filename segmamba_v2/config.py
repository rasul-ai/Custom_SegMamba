"""Shared settings for SegMamba v2 (Phase A: fixed training recipe).

Same data, split, patch size, augmentation and inference as 3_train.py / 4_predict.py.
What changes compared to the original SegMamba run:
  1. the LR schedule finishes (max_epoch = 250, so poly decays to 0)
  2. loss = BCE + soft Dice instead of 4-class CE
  3. region-based outputs: 3 sigmoid channels (TC, WT, ET) instead of 4-class softmax
  4. deep supervision on the decoder (1/2, 1/4, 1/8 resolution)
  5. ET post-processing, tuned on the validation split (postprocess.py)
  6. --seed, so the run can be repeated with 2-3 seeds
"""
import argparse
import os
import sys

# Scripts live in <repo>/segmamba_v2/ but use the repo's light_training, vendored monai
# and model_segmamba, with repo-relative paths (same trick as swinunetr/config.py).
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
os.chdir(ROOT)

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch

# Every script accepts --seed; parse_known_args leaves the script's own arguments alone.
_parser = argparse.ArgumentParser(add_help=False)
default_seed = 123  # seed of the original 3_train.py and swinunetr/train.py
_parser.add_argument("--seed", type=int, default=default_seed)
seed = _parser.parse_known_args()[0].seed

# ---------------- data (same as SegMamba) ----------------
data_dir = os.path.join(ROOT, "data", "fullres", "train")
raw_data_dir = os.path.join(ROOT, "data", "BraTS2020_TrainingData", "MICCAI_BraTS2020_TrainingData")
roi_size = [128, 128, 128]

# ---------------- training ----------------
run_name = "segmamba_v2" if seed == default_seed else f"segmamba_v2_seed{seed}"
logdir = os.path.join("logs", run_name)
model_save_path = os.path.join(logdir, "model")
split_file = os.path.join(logdir, "data_split.json")
# Splits saved by earlier runs. train.py refuses to start if its split differs from any that exists,
# because the split depends on the file-listing order of data_dir (glob), not only on random.seed(42).
reference_split_files = [os.path.join("logs", "swinunetr", "data_split.json"),
                         os.path.join("logs", "segmamba", "data_split.json")]
env = "pytorch"
max_epoch = 250  # the poly schedule is computed from this, so the LR reaches ~0 at the last epoch
batch_size = 2
val_every = 2
num_gpus = 1
device = "cuda:0"
augmentation = True

# SegMamba's original optimizer; it now anneals properly because max_epoch matches the real run length.
# "adamw" = AdamW lr 1e-4 + warmup + cosine, like the Swin UNETR run.
optimizer_name = "sgd"
sgd_lr = 1e-2
adamw_lr = 1e-4
adamw_weight_decay = 1e-5
warmup_ratio = 0.03

# ---------------- model / loss ----------------
feat_size = [48, 96, 192, 384]
depths = [2, 2, 2, 2]
num_regions = 3  # TC, WT, ET (sigmoid each)
deep_supervision = True
batch_dice = True  # Dice over the whole batch: more stable for small ET patches

# ---------------- inference (same as 4_predict.py) ----------------
sw_batch_size = 2
sw_overlap = 0.5
mirror_axes = [0, 1, 2]
threshold = 0.5
postprocess_file = os.path.join(logdir, "postprocessing.json")


def pred_dir(split, postprocessed=False):
    """prediction_results/<run>[_val][_pp]; the test folder name works with 5_compute_metrics.py."""
    name = run_name + ("_val" if split == "val" else "") + ("_pp" if postprocessed else "")
    return os.path.join("prediction_results", name)


def build_model():
    from segmamba_v2.model import SegMambaDS

    return SegMambaDS(in_chans=4,
                      out_chans=num_regions,
                      depths=depths,
                      feat_size=feat_size,
                      deep_supervision=deep_supervision)


def map_brats_labels(label):
    """BraTS2020 labels {0,1,2,4} -> {0,1,2,3}; padding (-1) -> 0. Same as 3_train.py."""
    label = torch.where(label == 4, torch.full_like(label, 3), label)
    label = torch.where(label < 0, torch.zeros_like(label), label)
    return label


def convert_labels(labels, dim=1):
    """TC, WT, ET region masks, same order as SegMamba."""
    result = [(labels == 1) | (labels == 3), (labels == 1) | (labels == 3) | (labels == 2), labels == 3]
    return torch.cat(result, dim=dim).float()


def regions_to_labels(regions, dim=1):
    """Thresholded TC/WT/ET masks -> label map {0,1,2,3}, written in nnU-Net's region order (WT, TC, ET).

    Converting back with convert_labels() guarantees ET ⊂ TC ⊂ WT.
    """
    tc, wt, et = regions.select(dim, 0).bool(), regions.select(dim, 1).bool(), regions.select(dim, 2).bool()
    labels = torch.zeros_like(tc, dtype=torch.uint8)
    labels[wt] = 2
    labels[tc] = 1
    labels[et] = 3
    return labels.unsqueeze(dim)


def split_names(ds):
    return sorted(os.path.basename(p)[:-4] for p in ds.datalist)
