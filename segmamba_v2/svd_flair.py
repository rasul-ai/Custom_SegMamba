"""Prepare SVD scans like MS3SEG: T1 registered to FLAIR, FLAIR brain mask; everything stays on the native FLAIR grid.

Same FSL commands as MS3SEG's preprocessing/coregistration_and_brain_extraction.py:
  flirt -in T1 -ref FLAIR -out T1_reg -searchrx -180 180 -searchry -180 180
  bet FLAIR FLAIR_brain -f 0.5 -m
Needs FSL (flirt, bet) on PATH.

    python segmamba_v2/svd_flair.py --inputs /path/to/data/LSTAI_inputs --out data/svd_flair
        -> data/svd_flair/SVD_###/{flair,t1,brain_mask}.nii.gz and data/svd_flair/qc.csv
Then predict.py --input_dir data/svd_flair writes label maps on the native FLAIR grid, i.e. the grid of
LST-AI's space-flair_seg-lst.nii.gz, so evaluate.py compares them directly.
"""
import config as cfg  # must be first

import argparse
import csv
import os
import shutil
import subprocess

import numpy as np
import SimpleITK as sitk

FSL_ENV = dict(os.environ, FSLOUTPUTTYPE="NIFTI_GZ")
# Brain mask volume. MS3SEG (bet on the same kind of 2D FLAIR): 1072-1850 ml. Outside this range bet most likely
# kept skull/neck or cut brain -> check the case visually. Older SVD brains are smaller, hence the lower bound.
BRAIN_ML_RANGE = (800, 2000)


def run(*command):
    subprocess.run([str(c) for c in command], check=True, env=FSL_ENV)


def check_fsl():
    missing = [tool for tool in ("flirt", "bet") if shutil.which(tool) is None]
    if missing:
        raise RuntimeError(f"FSL tools not found on PATH: {missing}")


def case_dirs(inputs):
    names = sorted(n for n in os.listdir(inputs) if os.path.isdir(os.path.join(inputs, n)))
    return [n for n in names if all(os.path.exists(os.path.join(inputs, n, f"{m.lower()}.nii.gz")) for m in cfg.modalities)]


def prepare(inputs, out):
    cases = case_dirs(inputs)
    if not cases:
        raise FileNotFoundError(f"no <case>/flair.nii.gz + t1.nii.gz folders in {inputs}")
    qc_rows = []
    for case in cases:
        case_out = os.path.join(out, case)
        os.makedirs(case_out, exist_ok=True)
        flair = os.path.join(case_out, "flair.nii.gz")
        shutil.copy2(os.path.join(inputs, case, "flair.nii.gz"), flair)
        run("flirt", "-in", os.path.join(inputs, case, "t1.nii.gz"), "-ref", flair,
            "-out", os.path.join(case_out, "t1.nii.gz"), "-searchrx", "-180", "180", "-searchry", "-180", "180")
        brain = os.path.join(case_out, "flair_brain")
        run("bet", flair, brain, "-f", "0.5", "-m")
        os.replace(f"{brain}_mask.nii.gz", os.path.join(case_out, "brain_mask.nii.gz"))
        os.remove(f"{brain}.nii.gz")  # predict.py applies the mask itself

        mask = sitk.ReadImage(os.path.join(case_out, "brain_mask.nii.gz"))
        brain_ml = float((sitk.GetArrayFromImage(mask) > 0).sum() * np.prod(mask.GetSpacing()) / 1000)
        flag = "" if BRAIN_ML_RANGE[0] <= brain_ml <= BRAIN_ML_RANGE[1] else "CHECK brain extraction"
        qc_rows.append({"case": case, "brain_ml": round(brain_ml, 1), "flag": flag})
        print(f"{case}: brain mask {brain_ml:.0f} ml {flag}")

    qc_file = os.path.join(out, "qc.csv")
    with open(qc_file, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(qc_rows[0]))
        writer.writeheader()
        writer.writerows(qc_rows)
    flagged = [r["case"] for r in qc_rows if r["flag"]]
    print(f"{len(qc_rows)} cases prepared in {out}; flagged: {flagged or 'none'} (see {qc_file}). "
          f"Also check a few T1-on-FLAIR overlays: the registration is not checked automatically.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", required=True, help="e.g. data/LSTAI_inputs (one folder per case with flair + t1)")
    parser.add_argument("--out", required=True)
    args, _ = parser.parse_known_args()

    check_fsl()
    prepare(cfg.user_path(args.inputs), cfg.user_path(args.out))
