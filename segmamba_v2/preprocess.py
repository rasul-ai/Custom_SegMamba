"""Preprocess MSLesSeg for training and build the patient-level folds.

Run:  python segmamba_v2/preprocess.py --mslesseg_dir "/path/to/MSLesSeg Dataset"
      (the folder that holds train/, test/ and info_dataset/)

Writes to data/mslesseg_fullres/:
  cases.json   every scan: patient, timepoint, split, FLAIR/T1/MASK paths, lesion volume, exclusion reason
  splits.json  official test cases + 5 patient-level folds (train/val) over the train patients,
               stratified by lesion load; scans in config.excluded are in no fold
  <case>.npz/.pkl  cropped, normalized FLAIR + T1 and the lesion mask of every fold scan (for train.py)

The images are already in MNI152 1 mm and skull-stripped, so preprocessing is: crop to the brain,
z-score each channel inside its brain mask (0 outside), record foreground locations for patch sampling.
predict.py runs the same FlairT1Preprocessor on every case it predicts.
"""
import config as cfg  # must be first: sets sys.path, cwd and CUDA allocator config

import argparse
import glob
import json
import os
import random
import re
from collections import defaultdict

import numpy as np
import SimpleITK as sitk

from light_training.preprocessing.preprocessors.preprocessor_mri import MultiModalityPreprocessor


class FlairT1Preprocessor(MultiModalityPreprocessor):
    """cases: {case_name: {"FLAIR": path, "T1": path, optional "MASK": path}}."""

    def __init__(self, cases):
        super().__init__(base_dir="", image_dir="", data_filenames=cfg.modalities, seg_filename="")
        self.cases = cases
        # run_case_npy() needs these even when run() is not called (prediction)
        self.out_spacing = cfg.out_spacing
        self.all_labels = [1]
        self.foreground_intensity_properties_per_channel = {}

    def get_iterable_list(self):
        return list(self.cases)

    def _normalize(self, data, seg, foreground_intensity_properties_per_channel):
        # z-score inside each channel's brain (nonzero) mask, 0 outside; robust to how much background the crop keeps
        for c in range(data.shape[0]):
            brain = data[c] != 0
            if brain.any():
                values = data[c][brain]
                data[c] = np.where(brain, (data[c] - values.mean()) / max(values.std(), 1e-8), 0)
        return data

    def read_data(self, case_name):
        files = self.cases[case_name]
        images = [sitk.ReadImage(files[m]) for m in cfg.modalities]
        ref = images[0]
        for m, image in zip(cfg.modalities[1:], images[1:]):
            if image.GetSize() != ref.GetSize() or not np.allclose(image.GetSpacing(), ref.GetSpacing(), atol=1e-3):
                raise ValueError(f"{case_name}: {m} grid {image.GetSize()} {image.GetSpacing()} differs from "
                                 f"{cfg.modalities[0]} {ref.GetSize()} {ref.GetSpacing()}; register them to the same space first")
        data = np.stack([sitk.GetArrayFromImage(image).astype(np.float32) for image in images])

        seg = None
        if files.get("MASK"):
            seg = (sitk.GetArrayFromImage(sitk.ReadImage(files["MASK"])) > 0).astype(np.float32)[None]

        properties = {"spacing": ref.GetSpacing(),
                      "raw_size": data.shape[1:],
                      "name": case_name}
        return data, seg, properties


def find_cases(mslesseg_dir):
    """{case: info} for train/P<n>/T<k>/P<n>_T<k>_* and test/P<n>/P<n>_*."""
    cases = {}
    for split in ("train", "test"):
        for flair in sorted(glob.glob(os.path.join(mslesseg_dir, split, "**", "*_FLAIR.nii.gz"), recursive=True)):
            case = os.path.basename(flair)[:-len("_FLAIR.nii.gz")]
            m = re.fullmatch(r"(P\d+)(?:_(T\d+))?", case)
            if m is None:
                print(f"SKIP {flair}: unexpected file name")
                continue
            prefix = flair[:-len("FLAIR.nii.gz")]
            info = {"patient": m.group(1), "timepoint": m.group(2) or "T1", "split": split}
            info.update({k: f"{prefix}{k}.nii.gz" for k in ("FLAIR", "T1", "MASK")})
            missing = [k for k in ("FLAIR", "T1", "MASK") if not os.path.exists(info[k])]
            if missing:
                print(f"SKIP {case}: missing {missing}")
                continue
            cases[case] = info
    return cases


def make_folds(patient_load, num_folds, seed):
    """Patients sorted by lesion load, dealt in shuffled blocks of num_folds -> similar load in every fold."""
    rng = random.Random(seed)
    patients = sorted(patient_load, key=lambda p: (patient_load[p], int(p[1:])))
    folds = [[] for _ in range(num_folds)]
    for i in range(0, len(patients), num_folds):
        order = list(range(num_folds))
        rng.shuffle(order)
        for patient, f in zip(patients[i:i + num_folds], order):
            folds[f].append(patient)
    return folds


def build_splits(cases):
    train_cases = [c for c, info in cases.items() if info["split"] == "train" and c not in cfg.excluded]
    by_patient = defaultdict(list)
    for c in train_cases:
        by_patient[cases[c]["patient"]].append(c)
    patient_load = {p: float(np.mean([cases[c]["lesion_ml"] for c in cs])) for p, cs in by_patient.items()}

    folds = []
    for val_patients in make_folds(patient_load, cfg.num_folds, cfg.split_seed):
        val = sorted(c for p in val_patients for c in by_patient[p])
        folds.append({"train": sorted(c for c in train_cases if c not in val), "val": val})
    return {"num_folds": cfg.num_folds,
            "seed": cfg.split_seed,
            "test": sorted(c for c, info in cases.items() if info["split"] == "test"),
            "excluded": {c: r for c, r in cfg.excluded.items() if c in cases},
            "folds": folds}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mslesseg_dir", required=True, help='the "MSLesSeg Dataset" folder (holds train/ and test/)')
    parser.add_argument("--num_processes", type=int, default=8)
    args, _ = parser.parse_known_args()

    cases = find_cases(cfg.user_path(args.mslesseg_dir))
    if not cases:
        raise FileNotFoundError(f"no *_FLAIR.nii.gz under {args.mslesseg_dir}/train or /test")
    for case, info in cases.items():
        mask = sitk.ReadImage(info["MASK"])
        info["lesion_ml"] = float((sitk.GetArrayFromImage(mask) > 0).sum() * np.prod(mask.GetSpacing()) / 1000)
        if case in cfg.excluded:
            info["excluded"] = cfg.excluded[case]

    splits = build_splits(cases)
    os.makedirs(cfg.data_dir, exist_ok=True)
    with open(cfg.cases_file, "w") as f:
        json.dump(cases, f, indent=1)
    with open(cfg.splits_file, "w") as f:
        json.dump(splits, f, indent=1)

    n_train = sum(info["split"] == "train" for info in cases.values())
    print(f"{len(cases)} scans: {n_train} train, {len(splits['test'])} test; excluded from folds: {sorted(splits['excluded'])}")
    for k, fold in enumerate(splits["folds"]):
        val_patients = sorted({cases[c]["patient"] for c in fold["val"]}, key=lambda p: int(p[1:]))
        load = np.median([cases[c]["lesion_ml"] for c in fold["val"]])
        print(f"fold {k}: train {len(fold['train'])} scans, val {len(fold['val'])} scans "
              f"({len(val_patients)} patients, median lesion load {load:.1f} ml)")

    # Unpacked .npy files of an earlier run would be reused as they are (dataset.py never overwrites them).
    for stale in glob.glob(os.path.join(cfg.data_dir, "*.npy")):
        os.remove(stale)

    fold_cases = sorted(set(splits["folds"][0]["train"]) | set(splits["folds"][0]["val"]))
    preprocessor = FlairT1Preprocessor({c: cases[c] for c in fold_cases})
    preprocessor.run(output_spacing=cfg.out_spacing,
                     output_dir=cfg.data_dir,
                     all_labels=[1],
                     num_processes=args.num_processes)
    print(f"saved {len(fold_cases)} cases to {cfg.data_dir}")
