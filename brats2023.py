"""BraTS2023 (GLI) settings shared by 2_preprocessing_mri.py, the three models (3_train.py / 4_predict.py,
segmamba_v2/, swinunetr/) and the metrics scripts.

Raw data: <raw_data_dir>/<case>/<case>-{t2f,t2w,t1n,t1c,seg}.nii.gz, read as they are (no renaming step).
Labels are already {0, 1 NCR, 2 ED, 3 ET} (BraTS2020 used 4 for ET).
Preprocessed data, logs and predictions get their own paths, so they never mix with the BraTS2020 runs of
branch main (data/fullres/train, logs/<model>, prediction_results/<model>) in the same checkout.
"""
import os

ROOT = os.path.dirname(os.path.abspath(__file__))

# set BRATS2023_DIR / BRATS2023_FULLRES_DIR to keep the data elsewhere (e.g. on a bigger disk)
raw_data_dir = os.environ.get("BRATS2023_DIR",
                              os.path.join(ROOT, "data", "BraTS2023", "ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData"))
data_dir = os.environ.get("BRATS2023_FULLRES_DIR", os.path.join(ROOT, "data", "fullres_brats2023", "train"))
modalities = ["t2f", "t2w", "t1n", "t1c"]  # FLAIR, T2, T1, T1ce: the channel order of the BraTS2020 runs
logs_root = os.path.join("logs", "brats2023")  # relative to the repo root, like the BraTS2020 logs
results_root = os.path.join("prediction_results", "brats2023")


def case_file(case_name, suffix):
    """<raw_data_dir>/<case>/<case>-<suffix>.nii.gz, e.g. suffix "t1c" or "seg"."""
    return os.path.join(raw_data_dir, case_name, f"{case_name}-{suffix}.nii.gz")


def patient_id(case_name):
    """BraTS-GLI-00008-001 -> "00008". The last number is the timepoint; 118 patients have two scans."""
    return case_name.split("-")[2]


def get_train_val_test_loader(data_dir=data_dir):
    """70/10/20 split by patient (seed 42), so both timepoints of a patient are always in the same split."""
    from light_training.dataloading.dataset import get_train_val_test_loader_from_train_by_group

    return get_train_val_test_loader_from_train_by_group(data_dir, patient_id)
