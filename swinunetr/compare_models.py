"""Compare SegMamba and Swin UNETR on the test split.

Needs both metric files produced by the repo's 5_compute_metrics.py:
    python 5_compute_metrics.py --pred_name segmamba
    python 5_compute_metrics.py --pred_name swinunetr
    python swinunetr/compare_models.py
"""
import argparse
import os

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REGIONS = ["TC", "WT", "ET"]


def load(name, metrics_dir):
    path = os.path.join(metrics_dir, f"{name}.npy")
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} not found; run: python 5_compute_metrics.py --pred_name {name}")
    return np.load(path)  # (num_cases, 3 regions, [dice, hd95])


def wilcoxon_p(a, b):
    try:
        from scipy.stats import wilcoxon
        return wilcoxon(a, b).pvalue if np.any(a != b) else 1.0
    except ImportError:
        return float("nan")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs=2, default=["segmamba", "swinunetr"])
    parser.add_argument("--metrics_dir", default=os.path.join(ROOT, "prediction_results", "result_metrics"))
    args = parser.parse_args()

    a_name, b_name = args.models
    a, b = load(a_name, args.metrics_dir), load(b_name, args.metrics_dir)
    if a.shape != b.shape:
        raise ValueError(f"{a_name} has {a.shape[0]} cases but {b_name} has {b.shape[0]}; they must share the test split")
    print(f"{a.shape[0]} test cases\n")

    rows = []
    for m, metric in enumerate(["Dice", "HD95"]):
        for r, region in enumerate(REGIONS + ["Mean"]):
            if region == "Mean":
                va, vb = a[:, :, m].mean(axis=1), b[:, :, m].mean(axis=1)
            else:
                va, vb = a[:, r, m], b[:, r, m]
            rows.append((f"{metric} {region}", va, vb))

    header = f"{'metric':<11}{a_name:>20}{b_name:>20}{'wilcoxon p':>13}"
    print(header)
    print("-" * len(header))
    for label, va, vb in rows:
        print(f"{label:<11}{va.mean():>12.4f} ± {va.std():<5.3f}{vb.mean():>12.4f} ± {vb.std():<5.3f}{wilcoxon_p(va, vb):>13.4g}")
    print("\nDice: higher is better. HD95 (mm): lower is better; cases with an empty prediction or GT get HD95 = 50.")
