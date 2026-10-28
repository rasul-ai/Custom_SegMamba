"""Segmentation metrics per case and class, summarized over cases.

MS3SEG (predictions from predict.py, reference = the expert masks):
    python segmamba_v2/evaluate.py --fold 1 --split val --tune   # pick the minimum MS lesion size on validation
    python segmamba_v2/evaluate.py --fold 1 --split test
Any folder of predicted label maps against binary reference masks (e.g. SVD: SegMamba vs LST-AI, both on the
native FLAIR grid). The predicted labels in --pred_labels are merged into one mask first (default 2 3 = normal WMH +
MS lesion; use --pred_labels 3 for MS lesions only):
    python segmamba_v2/evaluate.py --pred_dir prediction_results/segmamba_v2_ms3seg_fold1/svd_flair \
        --gt_dir /path/to/output/LSTAI_results --gt_file space-flair_seg-lst.nii.gz

Metrics: Dice, reference and predicted volume (ml) and their absolute difference, HD95 (mm). For MS lesions (and
for the merged comparison) also lesion-wise sensitivity (share of reference lesions that the prediction touches),
precision (share of predicted lesions that touch a reference lesion) and F1; lesions are 26-connected components,
"touch" = at least one shared voxel. Over cases: mean, SD, median and IQR per class, and volume agreement
(Pearson r, ICC(2,1), Bland-Altman). Writes metrics.csv into the prediction folder.
"""
import config as cfg  # must be first

import argparse
import csv
import json
import os

import numpy as np
import SimpleITK as sitk
from scipy import ndimage
from tqdm import tqdm

from predict import remove_small_lesions

MIN_SIZE_GRID = [0, 3, 5, 10, 20, 30, 50]
COLUMNS = ["case", "class", "dice", "lesion_sensitivity", "lesion_precision", "lesion_f1", "n_lesions_ref",
           "n_lesions_pred", "vol_ref_ml", "vol_pred_ml", "abs_vol_diff_ml", "hd95_mm"]
STRUCTURE = np.ones((3, 3, 3))


def load_image(path):
    image = sitk.ReadImage(path)
    return sitk.GetArrayFromImage(image), image.GetSpacing()[::-1]  # spacing in array (z, y, x) order


def dice_score(pred, gt):
    total = pred.sum() + gt.sum()
    return 1.0 if total == 0 else 2.0 * (pred & gt).sum() / total


def lesion_scores(pred, gt):
    """(sensitivity, precision, F1, n reference lesions, n predicted lesions); undefined values are NaN."""
    gt_labels, n_gt = ndimage.label(gt, structure=STRUCTURE)
    pred_labels, n_pred = ndimage.label(pred, structure=STRUCTURE)
    hit_gt = len(np.setdiff1d(np.unique(gt_labels[pred]), [0]))
    hit_pred = len(np.setdiff1d(np.unique(pred_labels[gt]), [0]))
    sens = hit_gt / n_gt if n_gt else np.nan
    prec = hit_pred / n_pred if n_pred else np.nan
    if n_gt == 0 and n_pred == 0:
        f1 = np.nan
    elif n_gt == 0 or n_pred == 0 or sens + prec == 0:
        f1 = 0.0
    else:
        f1 = 2 * sens * prec / (sens + prec)
    return sens, prec, f1, n_gt, n_pred


def case_metrics(case, name, pred, gt, spacing, lesionwise, with_hd95=True):
    if pred.shape != gt.shape:
        raise ValueError(f"{case}: prediction {pred.shape} and reference {gt.shape} are on different grids")
    voxel_ml = float(np.prod(spacing)) / 1000
    sens, prec, f1, n_gt, n_pred = lesion_scores(pred, gt) if lesionwise else (np.nan,) * 5
    hd95 = np.nan
    if with_hd95 and pred.any() and gt.any():
        from medpy import metric

        hd95 = metric.binary.hd95(pred, gt, voxelspacing=spacing)
    vol_gt, vol_pred = gt.sum() * voxel_ml, pred.sum() * voxel_ml
    return {"case": case, "class": name, "dice": dice_score(pred, gt), "lesion_sensitivity": sens,
            "lesion_precision": prec, "lesion_f1": f1, "n_lesions_ref": n_gt, "n_lesions_pred": n_pred,
            "vol_ref_ml": vol_gt, "vol_pred_ml": vol_pred, "abs_vol_diff_ml": abs(vol_pred - vol_gt), "hd95_mm": hd95}


def icc_2_1(x, y):
    """ICC(2,1): two-way random effects, absolute agreement, single measurement (Shrout & Fleiss)."""
    data = np.stack([x, y], axis=1)
    n, k = data.shape
    grand = data.mean()
    ms_rows = k * ((data.mean(axis=1) - grand) ** 2).sum() / (n - 1)
    ms_cols = n * ((data.mean(axis=0) - grand) ** 2).sum() / (k - 1)
    resid = data - data.mean(axis=1, keepdims=True) - data.mean(axis=0, keepdims=True) + grand
    ms_err = (resid ** 2).sum() / ((n - 1) * (k - 1))
    return (ms_rows - ms_err) / (ms_rows + (k - 1) * ms_err + k * (ms_cols - ms_err) / n)


def summarize(rows):
    for name in dict.fromkeys(r["class"] for r in rows):
        class_rows = [r for r in rows if r["class"] == name]
        print(f"\n{name}: {len(class_rows)} cases")
        print(f"{'metric':<20}{'mean':>9}{'SD':>9}{'median':>9}{'IQR':>19}{'n':>5}")
        for col in COLUMNS[2:]:
            v = np.array([r[col] for r in class_rows], dtype=float)
            v = v[~np.isnan(v)]
            if len(v) == 0:
                continue
            q1, q3 = np.percentile(v, [25, 75])
            print(f"{col:<20}{v.mean():>9.3f}{v.std():>9.3f}{np.median(v):>9.3f}{f'{q1:.3f}-{q3:.3f}':>19}{len(v):>5}")

        ref = np.array([r["vol_ref_ml"] for r in class_rows])
        pred = np.array([r["vol_pred_ml"] for r in class_rows])
        if len(class_rows) >= 3 and ref.std() > 0 and pred.std() > 0:
            diff = pred - ref
            print(f"volume agreement (ml): Pearson r {np.corrcoef(ref, pred)[0, 1]:.3f}, ICC(2,1) {icc_2_1(ref, pred):.3f}, "
                  f"Bland-Altman bias {diff.mean():+.2f}, 95% limits {diff.mean() - 1.96 * diff.std(ddof=1):+.2f} "
                  f"to {diff.mean() + 1.96 * diff.std(ddof=1):+.2f}")

    names = list(dict.fromkeys(r["class"] for r in rows))
    if len(names) > 1:
        per_class = [np.mean([r["dice"] for r in rows if r["class"] == n]) for n in names]
        print(f"\nmean Dice over classes: {np.mean(per_class):.4f} ({', '.join(f'{n} {d:.4f}' for n, d in zip(names, per_class))})")


def write_csv(rows, path):
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"saved {path}")


def ms3seg_pairs(split):
    """[(case, prediction path, {label: reference mask path})] for an MS3SEG split of predict.py."""
    cases = cfg.load_json(cfg.cases_file)
    pred_dir = cfg.pred_dir(split)
    return pred_dir, [(c, os.path.join(pred_dir, f"{c}.nii.gz"), cases[c]["MASKS"]) for c in cfg.split_cases(split)]


def folder_pairs(pred_dir, gt_dir, gt_file):
    """[(case, prediction path, reference path)] for every <pred_dir>/<case>.nii.gz with <gt_dir>/<case>/<gt_file>."""
    pairs = []
    for name in sorted(os.listdir(pred_dir)):
        if not name.endswith(".nii.gz") or "_prob_" in name:
            continue
        case = name[:-len(".nii.gz")]
        gt = os.path.join(gt_dir, case, gt_file)
        if not os.path.exists(gt):
            print(f"SKIP {case}: no reference {gt}")
            continue
        pairs.append((case, os.path.join(pred_dir, name), gt))
    return pairs


def tune(pairs, pred_dir):
    info_file = os.path.join(pred_dir, "prediction_info.json")
    if os.path.exists(info_file) and cfg.load_json(info_file)["min_size"] > 0:
        raise RuntimeError(f"{pred_dir} was predicted with small lesions already removed; delete "
                           f"{cfg.postprocess_file}, rerun predict.py --split val, then tune")
    loaded = []
    for case, pred_path, masks in tqdm(pairs, desc="loading"):
        pred = load_image(pred_path)[0] == cfg.lesion_label
        loaded.append((pred, load_image(masks[str(cfg.lesion_label)])[0] > 0))

    results = []
    for min_size in MIN_SIZE_GRID:
        dices, f1s = [], []
        for pred, gt in loaded:
            pred, _ = remove_small_lesions(pred, min_size)
            dices.append(dice_score(pred, gt))
            f1s.append(lesion_scores(pred, gt)[2])
        results.append({"min_size": min_size, "dice": float(np.mean(dices)), "lesion_f1": float(np.nanmean(f1s))})

    # Highest lesion F1 (Dice barely moves when small lesions go); on ties the smallest size.
    best = max(results, key=lambda r: (round(r["lesion_f1"], 4), -r["min_size"]))
    print(f"\nMS lesion\n{'min_size':>9}{'Dice':>9}{'lesion F1':>11}")
    for r in results:
        print(f"{r['min_size']:>9}{r['dice']:>9.4f}{r['lesion_f1']:>11.4f}" + ("   <- chosen" if r is best else ""))
    with open(cfg.postprocess_file, "w") as f:
        json.dump({"tuned_on": f"fold {cfg.fold} val", "min_size": best["min_size"], "grid": results}, f, indent=1)
    print(f"saved {cfg.postprocess_file}; predict.py now removes predicted MS lesions < {best['min_size']} voxels")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=["val", "test"], help="MS3SEG split predicted by predict.py --split")
    parser.add_argument("--tune", action="store_true", help="with --split val: choose the minimum MS lesion size")
    parser.add_argument("--pred_dir", help="instead of --split: folder with <case>.nii.gz label maps from predict.py")
    parser.add_argument("--pred_labels", type=int, nargs="+", default=[2, 3],
                        help="with --pred_dir: predicted labels merged into one mask (default: normal WMH + MS lesion)")
    parser.add_argument("--gt_dir", help="with --pred_dir: folder with <case>/<gt_file>")
    parser.add_argument("--gt_file", default="space-flair_seg-lst.nii.gz")
    parser.add_argument("--no_hd95", action="store_true", help="skip HD95 (the slowest metric)")
    args, _ = parser.parse_known_args()

    if args.split:
        pred_dir, pairs = ms3seg_pairs(args.split)
    elif args.pred_dir and args.gt_dir:
        if not set(args.pred_labels) <= set(cfg.classes):
            parser.error(f"--pred_labels must be among {cfg.classes}")
        pred_dir = cfg.user_path(args.pred_dir)
        pairs = folder_pairs(pred_dir, cfg.user_path(args.gt_dir), args.gt_file)
    else:
        parser.error("give --split, or --pred_dir and --gt_dir")
    if not pairs:
        raise FileNotFoundError("no prediction/reference pairs found")

    if args.tune:
        if args.split != "val":
            parser.error("--tune only with --split val")
        tune(pairs, pred_dir)
    else:
        rows = []
        for case, pred_path, reference in tqdm(pairs, desc="cases"):
            label_map, _ = load_image(pred_path)
            if args.split:  # one row per class, against that class's expert mask
                for label, name in cfg.classes.items():
                    gt, spacing = load_image(reference[str(label)])
                    rows.append(case_metrics(case, name, label_map == label, gt > 0, spacing,
                                             lesionwise=label == cfg.lesion_label, with_hd95=not args.no_hd95))
            else:  # merged predicted labels against one binary reference
                gt, spacing = load_image(reference)
                name = "+".join(cfg.classes[label] for label in args.pred_labels)
                rows.append(case_metrics(case, name, np.isin(label_map, args.pred_labels), gt > 0, spacing,
                                         lesionwise=True, with_hd95=not args.no_hd95))
        summarize(rows)
        write_csv(rows, os.path.join(pred_dir, "metrics.csv"))
