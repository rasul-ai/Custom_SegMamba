"""Dice / HD95 for TC, WT, ET on the val or test split, under two conventions.

  legacy       : identical to 5_compute_metrics.py; if the prediction OR the ground truth is empty -> Dice 0, HD95 50.
                 Used in the Sep 30 report, so use it for comparisons with those numbers.
  empty_aware  : if BOTH are empty -> Dice 1, HD95 0 (the case was predicted correctly); one empty -> Dice 0, HD95 50.
                 Only this convention can reward ET post-processing on cases without enhancing tumor.

Run:  python segmamba_v2/evaluate.py --pred_name segmamba_v2        --split test
      python segmamba_v2/evaluate.py --pred_name segmamba_v2_pp     --split test
      python segmamba_v2/evaluate.py --pred_name segmamba_v2_val    --split val
Writes prediction_results/result_metrics/<pred_name>.npy (legacy, same format as 5_compute_metrics.py,
so swinunetr/compare_models.py can read it) and <pred_name>_empty_aware.npy.
"""
import config as cfg  # must be first

import argparse
import os

import numpy as np
import SimpleITK as sitk
import torch
from tqdm import tqdm

from light_training.dataloading.dataset import get_train_val_test_loader_from_train

REGIONS = ["TC", "WT", "ET"]
METRICS_DIR = os.path.join("prediction_results", "result_metrics")


def case_names(split):
    train_ds, val_ds, test_ds = get_train_val_test_loader_from_train(cfg.data_dir)
    ds = val_ds if split == "val" else test_ds
    # same order as 5_compute_metrics.py iterates the dataset
    return [os.path.basename(p)[:-4] for p in ds.datalist]


def resolve_existing_path(candidates):
    for path in candidates:
        if os.path.exists(path):
            return path
    raise FileNotFoundError(f"None of these paths exists: {candidates}")


def load_gt(case_name):
    """(3, D, H, W) TC/WT/ET masks from the raw BraTS segmentation, same as 5_compute_metrics.py."""
    case_dir = os.path.join(cfg.raw_data_dir, case_name)
    path = resolve_existing_path([os.path.join(case_dir, f) for f in
                                  ("seg.nii.gz", "seg.nii", f"{case_name}_seg.nii.gz", f"{case_name}_seg.nii")])
    gt = sitk.GetArrayFromImage(sitk.ReadImage(path)).astype(np.int32)
    gt[gt == 4] = 3
    return cfg.convert_labels(torch.from_numpy(gt)[None], dim=0).numpy().astype(bool)


def pred_path(pred_name, case_name):
    folder = os.path.join("prediction_results", pred_name)
    return resolve_existing_path([os.path.join(folder, f"{case_name}.nii.gz"), os.path.join(folder, f"{case_name}.nii")])


def load_pred(pred_name, case_name):
    return sitk.GetArrayFromImage(sitk.ReadImage(pred_path(pred_name, case_name))).astype(bool)


def region_metrics(pred, gt):
    """Returns (legacy [dice, hd95], empty_aware [dice, hd95]) for one region."""
    from medpy import metric

    if pred.any() and gt.any():
        m = np.array([metric.binary.dc(pred, gt), metric.binary.hd95(pred, gt, voxelspacing=[1, 1, 1])])
        return m, m
    legacy = np.array([0.0, 50.0])
    empty_aware = np.array([1.0, 0.0]) if not pred.any() and not gt.any() else legacy
    return legacy, empty_aware


def summarize(name, results):
    print(f"\n{name}")
    print(f"{'':<6}{'Dice':>18}{'HD95 (mm)':>20}")
    for r, region in enumerate(REGIONS):
        d, h = results[:, r, 0], results[:, r, 1]
        print(f"{region:<6}{d.mean():>10.4f} ± {d.std():<6.3f}{h.mean():>12.3f} ± {h.std():<6.3f}")
    d, h = results[:, :, 0].mean(axis=1), results[:, :, 1].mean(axis=1)
    print(f"{'Mean':<6}{d.mean():>10.4f} ± {d.std():<6.3f}{h.mean():>12.3f} ± {h.std():<6.3f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred_name", required=True, help="folder name under prediction_results/")
    parser.add_argument("--split", choices=["val", "test"], default="test")
    args, _ = parser.parse_known_args()

    names = case_names(args.split)
    legacy = np.zeros((len(names), 3, 2), dtype=np.float32)
    empty_aware = np.zeros_like(legacy)
    n_no_et = 0
    for i, case_name in enumerate(tqdm(names)):
        gt, pred = load_gt(case_name), load_pred(args.pred_name, case_name)
        n_no_et += not gt[2].any()
        for r in range(3):
            legacy[i, r], empty_aware[i, r] = region_metrics(pred[r], gt[r])

    os.makedirs(METRICS_DIR, exist_ok=True)
    np.save(os.path.join(METRICS_DIR, f"{args.pred_name}.npy"), legacy)
    np.save(os.path.join(METRICS_DIR, f"{args.pred_name}_empty_aware.npy"), empty_aware)

    print(f"{len(names)} {args.split} cases, {n_no_et} without enhancing tumor in the ground truth")
    summarize("legacy convention (= 5_compute_metrics.py)", legacy)
    summarize("empty-aware convention", empty_aware)
