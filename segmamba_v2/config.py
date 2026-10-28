"""Shared settings for SegMamba v2 on MS3SEG (FLAIR + T1 -> ventricles, normal WMH, MS lesions).

Same network and training recipe as SegMamba v2 on BraTS (SGD + poly over 250 epochs, BCE + soft Dice,
deep supervision, 128^3 patches, sliding window + mirror TTA). What changes for MS3SEG:
  1. input = FLAIR + T1 only (2 channels), so the model also runs on data without T2 (e.g. the SVD cohort)
  2. output = 3 sigmoid channels: ventricles, normal WMH, abnormal WMH (MS lesion), as in MS3SEG's 4-class scenario
  3. the official MS3SEG split: 5 folds (64 train / 16 val patients) + 20 held-out test patients
  4. native FLAIR space (MS3SEG has no MNI step): skull-stripped with the dataset's brain mask, slices resampled to
     1 mm for the network, predictions resampled back onto the native FLAIR grid
  5. intensity z-score inside the brain mask
  6. post-processing = removal of predicted MS lesions below a size tuned on validation (evaluate.py --tune)
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
_parser.add_argument("--fold", type=int, default=1)  # official MS3SEG folds are numbered 1-5
_args = _parser.parse_known_args()[0]
seed, fold = _args.seed, _args.fold

# ---------------- data ----------------
modalities = ["FLAIR", "T1"]  # channel order; external cases provide flair.nii.gz, t1.nii.gz, brain_mask.nii.gz
# Label values follow MS3SEG (0 = background); output channel k predicts label k + 1.
classes = {1: "ventricles", 2: "nwmh", 3: "lesion"}
mask_folders = {1: "Vent", 2: "nWMH", 3: "abWMH"}  # MS_100_patient_masks/<folder>_Masks/<id>/<id>_<folder>_Mask.nii.gz
lesion_label = 3
num_classes = len(classes)
# written by preprocess.py; set MS3SEG_DATA_DIR to keep it elsewhere (e.g. on a bigger disk)
data_dir = os.environ.get("MS3SEG_DATA_DIR", os.path.join(ROOT, "data", "ms3seg_fullres"))
cases_file = os.path.join(data_dir, "cases.json")
splits_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ms3seg_splits.json")  # official MS3SEG split
# Spacing the network sees, in array (z, y, x) order. MS3SEG is 6-8 mm slices at 0.8984 mm in-plane: slices are
# resampled to 1 mm, in-plane stays native. (Resampling in-plane to 1 mm and back cost normal WMH ~7% Dice even for
# a perfect prediction; the z round trip is lossless.)
out_spacing = [1.0, 0.8984, 0.8984]
roi_size = [128, 128, 128]

# ---------------- training ----------------
run_name = f"segmamba_v2_ms3seg_fold{fold}" + ("" if seed == default_seed else f"_seed{seed}")
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
    """Patient IDs of the "train" or "val" split of the current fold, or of the official "test" split."""
    splits = load_json(splits_file)
    if split == "test":
        return sorted(splits["test_patients"])
    folds = splits["fold_assignments"]
    if f"fold_{fold}" not in folds:
        raise ValueError(f"--fold must be one of {sorted(int(k.split('_')[1]) for k in folds)}")
    return sorted(folds[f"fold_{fold}"][f"{split}_patients"])


def build_model():
    from segmamba_v2.model import SegMambaDS

    return SegMambaDS(in_chans=len(modalities),
                      out_chans=num_classes,
                      depths=depths,
                      feat_size=feat_size,
                      deep_supervision=deep_supervision)
