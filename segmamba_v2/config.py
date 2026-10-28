"""Shared settings for SegMamba v2 on MSLesSeg (MS lesion segmentation, FLAIR + T1).

Same network and training recipe as SegMamba v2 on BraTS (SGD + poly over 250 epochs, BCE + soft Dice,
deep supervision, 128^3 patches, sliding window + mirror TTA). What changes for MS lesions:
  1. input = FLAIR + T1 only (2 channels), so the model also runs on data without T2 (e.g. the SVD cohort)
  2. output = 1 sigmoid channel (lesion)
  3. patient-level 5-fold CV over the 53 MSLesSeg train patients; the official test set (22 patients) is held out
  4. intensity z-score inside each channel's brain mask
  5. scans with a broken or misregistered FLAIR/T1 are left out of training and validation (`excluded`)
  6. post-processing = removal of predicted lesions below a size tuned on validation (evaluate.py --tune)
"""
import argparse
import json
import os
import sys

# Scripts live in <repo>/segmamba_v2/ but use the repo's light_training, vendored monai
# and model_segmamba, with repo-relative paths.
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
launch_dir = os.getcwd()  # paths given on the command line are relative to this, not to ROOT
os.chdir(ROOT)

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

# Every script accepts --seed and --fold; parse_known_args leaves the script's own arguments alone.
_parser = argparse.ArgumentParser(add_help=False)
default_seed = 123
_parser.add_argument("--seed", type=int, default=default_seed)
_parser.add_argument("--fold", type=int, default=0)
_args = _parser.parse_known_args()[0]
seed, fold = _args.seed, _args.fold

# ---------------- data ----------------
modalities = ["FLAIR", "T1"]  # channel order; external cases provide flair.nii.gz and t1.nii.gz
# written by preprocess.py; set MSLESSEG_DATA_DIR to keep it elsewhere (e.g. on a bigger disk)
data_dir = os.environ.get("MSLESSEG_DATA_DIR", os.path.join(ROOT, "data", "mslesseg_fullres"))
cases_file = os.path.join(data_dir, "cases.json")
splits_file = os.path.join(data_dir, "splits.json")
num_folds = 5
split_seed = 12345
out_spacing = [1.0, 1.0, 1.0]  # MSLesSeg is already MNI152 1 mm; anything else is resampled
roi_size = [128, 128, 128]
# Found by QC of the preprocessed MSLesSeg data. The lesion masks follow FLAIR in all of these.
excluded = {
    "P49_T2": "T1 is empty",
    "P44_T1": "FLAIR registered about 1.6x too large (does not match T1)",
    "P45_T1": "FLAIR registered about 1.7x too large (does not match T1)",
    "P53_T1": "FLAIR registered about 1.6x too large (does not match T1)",
    "P12_T1": "T1 misaligned with FLAIR",
    "P12_T3": "T1 misaligned with FLAIR",
    "P12_T4": "T1 misaligned with FLAIR",
}

# ---------------- training ----------------
run_name = f"segmamba_v2_ms_fold{fold}" + ("" if seed == default_seed else f"_seed{seed}")
logdir = os.path.join("logs", run_name)
model_save_path = os.path.join(logdir, "model")
env = "pytorch"
max_epoch = 250  # the poly schedule is computed from this, so the LR reaches ~0 at the last epoch
batch_size = 2
val_every = 2
num_gpus = 1
device = "cuda:0"
augmentation = True
train_process = 18  # augmentation worker processes

optimizer_name = "sgd"  # "adamw" = AdamW lr 1e-4 + warmup + cosine
sgd_lr = 1e-2
adamw_lr = 1e-4
adamw_weight_decay = 1e-5
warmup_ratio = 0.03

# ---------------- model / loss ----------------
feat_size = [48, 96, 192, 384]
depths = [2, 2, 2, 2]
num_classes = 1  # lesion (sigmoid)
deep_supervision = True
batch_dice = True  # Dice over the whole batch: more stable for patches with few lesion voxels

# ---------------- inference ----------------
sw_batch_size = 2
sw_overlap = 0.5
mirror_axes = [0, 1, 2]
threshold = 0.5
postprocess_file = os.path.join(logdir, "postprocessing.json")


def pred_dir(name):
    """prediction_results/<run>/<name>, e.g. name = "val", "test" or the input folder name."""
    return os.path.join("prediction_results", run_name, name)


def user_path(path):
    """Absolute path for a command-line argument, resolved against the directory the script was started from."""
    return os.path.abspath(os.path.join(launch_dir, os.path.expanduser(path)))


def load_json(path):
    with open(path) as f:
        return json.load(f)


def split_cases(split):
    """Case names of the MSLesSeg "val" split (of the current fold) or of the official "test" split."""
    if not os.path.exists(splits_file):
        raise FileNotFoundError(f"{splits_file} not found; run segmamba_v2/preprocess.py first")
    splits = load_json(splits_file)
    if not 0 <= fold < len(splits["folds"]):
        raise ValueError(f"--fold must be 0..{len(splits['folds']) - 1}")
    return splits["folds"][fold][split] if split in ("train", "val") else splits["test"]


def build_model():
    from segmamba_v2.model import SegMambaDS

    return SegMambaDS(in_chans=len(modalities),
                      out_chans=num_classes,
                      depths=depths,
                      feat_size=feat_size,
                      deep_supervision=deep_supervision)
