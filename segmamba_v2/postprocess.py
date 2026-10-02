"""ET post-processing: remove small ET components, and relabel all ET as necrosis if too little ET is left.

Small false-positive ET blobs are common in cases without enhancing tumor. Relabelling ET as necrosis
removes it from the ET channel but keeps it in TC/WT, so only the ET region changes.

1) tune on the validation split (writes logs/segmamba_v2/postprocessing.json):
       python segmamba_v2/predict.py --split val
       python segmamba_v2/postprocess.py tune
2) apply to the test predictions (prediction_results/segmamba_v2 -> prediction_results/segmamba_v2_pp):
       python segmamba_v2/postprocess.py apply
"""
import config as cfg  # must be first

import argparse
import json
import os

import numpy as np
import SimpleITK as sitk
from skimage import measure
from tqdm import tqdm

from evaluate import case_names, load_gt, load_pred, pred_path

ET = 2  # channel index in TC/WT/ET
MIN_COMPONENT_GRID = [0, 10, 25, 50, 100, 200]
MIN_TOTAL_GRID = [0, 50, 100, 200, 300, 500, 750, 1000]


def postprocess_et(seg, min_component, min_total):
    """seg: (3, D, H, W) TC/WT/ET masks. Returns a copy with the ET channel cleaned (TC/WT unchanged)."""
    seg = seg.copy()
    et = seg[ET].astype(bool)
    if min_component > 0 and et.any():
        components = measure.label(et, connectivity=1)
        sizes = np.bincount(components.ravel())
        small = np.flatnonzero(sizes < min_component)
        et[np.isin(components, small[small > 0])] = False
    if et.sum() < min_total:
        et[:] = False
    seg[ET] = et
    return seg


def et_stats(pred_et, gt_et):
    """Per ET component of the prediction: (size, overlap with GT); plus the GT volume."""
    components = measure.label(pred_et, connectivity=1)
    sizes = np.bincount(components.ravel())[1:]
    overlaps = np.bincount(components[gt_et].ravel(), minlength=len(sizes) + 1)[1:]
    return sizes, overlaps, int(gt_et.sum())


def et_dice(sizes, overlaps, gt_sum, min_component, min_total):
    """Empty-aware ET Dice after post-processing, computed from component statistics (no re-labelling)."""
    keep = sizes >= min_component
    pred_sum, inter = sizes[keep].sum(), overlaps[keep].sum()
    if pred_sum < min_total:
        pred_sum, inter = 0, 0
    if pred_sum == 0 and gt_sum == 0:
        return 1.0
    return 2 * inter / (pred_sum + gt_sum)


def tune(pred_name):
    stats = [et_stats(load_pred(pred_name, c)[ET], load_gt(c)[ET]) for c in tqdm(case_names("val"), desc="val cases")]

    results = []
    for mc in MIN_COMPONENT_GRID:
        for mt in MIN_TOTAL_GRID:
            dices = [et_dice(*s, mc, mt) for s in stats]
            results.append({"min_component": mc, "min_total": mt, "et_dice": float(np.mean(dices))})

    baseline = results[0]["et_dice"]  # (0, 0) = no post-processing
    # Highest ET Dice; on ties prefer the mildest setting (the grid is ordered mild -> strong).
    best = max(results, key=lambda r: (round(r["et_dice"], 6), -r["min_component"], -r["min_total"]))
    if best["et_dice"] <= baseline:
        best = results[0]

    print(f"{len(stats)} validation cases, {sum(s[2] == 0 for s in stats)} without enhancing tumor")
    print(f"ET Dice (empty-aware) without post-processing: {baseline:.4f}")
    print(f"best: min_component={best['min_component']} min_total={best['min_total']} -> ET Dice {best['et_dice']:.4f}")
    print("top 5:")
    for r in sorted(results, key=lambda r: -r["et_dice"])[:5]:
        print(f"  min_component={r['min_component']:<4} min_total={r['min_total']:<5} ET Dice {r['et_dice']:.4f}")

    with open(cfg.postprocess_file, "w") as f:
        json.dump({"tuned_on": pred_name, "baseline_et_dice": baseline,
                   "min_component": best["min_component"], "min_total": best["min_total"],
                   "best_et_dice": best["et_dice"], "grid": results}, f, indent=1)
    print(f"saved {cfg.postprocess_file}")


def apply(pred_name, out_name):
    with open(cfg.postprocess_file) as f:
        params = json.load(f)
    mc, mt = params["min_component"], params["min_total"]
    out_dir = os.path.join("prediction_results", out_name)
    os.makedirs(out_dir, exist_ok=True)
    print(f"min_component={mc} min_total={mt}: prediction_results/{pred_name} -> {out_dir}")

    changed = 0
    for case_name in tqdm(case_names("test"), desc="test cases"):
        image = sitk.ReadImage(pred_path(pred_name, case_name))
        seg = sitk.GetArrayFromImage(image)
        new_seg = postprocess_et(seg, mc, mt)
        changed += not np.array_equal(seg, new_seg)
        out = sitk.GetImageFromArray(new_seg.astype(np.uint8))
        out.CopyInformation(image)
        sitk.WriteImage(out, os.path.join(out_dir, f"{case_name}.nii.gz"))
    print(f"ET changed in {changed} cases")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["tune", "apply"])
    parser.add_argument("--pred_name", default=None, help="default: <run>_val for tune, <run> for apply")
    parser.add_argument("--out_name", default=None, help="apply only; default: <run>_pp")
    args, _ = parser.parse_known_args()

    if args.mode == "tune":
        tune(args.pred_name or os.path.basename(cfg.pred_dir("val")))
    else:
        apply(args.pred_name or os.path.basename(cfg.pred_dir("test")),
              args.out_name or os.path.basename(cfg.pred_dir("test", postprocessed=True)))
