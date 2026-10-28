"""Bring SVD scans into the training space (MNI152 1 mm, skull-stripped) and predictions back to native FLAIR space.

Same steps as the MSLesSeg preprocessing (MSLesSeg-2024-main/registration.py): FSL flirt of FLAIR and of T1 to the
MNI152 1 mm template (12 DOF, flirt defaults), then bet on each registered image. Needs FSL (flirt, bet, convert_xfm,
fslmaths) on PATH; the template defaults to $FSLDIR/data/standard/MNI152_T1_1mm.nii.gz.

1) python segmamba_v2/svd_mni.py to_mni --inputs /path/to/data/LSTAI_inputs --out data/svd_mni
       -> data/svd_mni/SVD_###/{flair,t1}.nii.gz + {flair,t1}_to_mni.mat, and data/svd_mni/qc.csv
2) python segmamba_v2/predict.py --fold 0 --input_dir data/svd_mni
3) python segmamba_v2/svd_mni.py to_native --inputs /path/to/data/LSTAI_inputs --mni_dir data/svd_mni \
       --pred_dir prediction_results/segmamba_v2_ms_fold0/svd_mni --out prediction_results/segmamba_v2_ms_fold0/svd_native
       -> svd_native/SVD_###/space-flair_prob-segmamba.nii.gz and space-flair_seg-segmamba.nii.gz on the grid of
          LSTAI_inputs/SVD_###/flair.nii.gz, i.e. the same grid as LST-AI's space-flair_seg-lst.nii.gz
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
# FLAIR/T1 brain-volume ratio after bet. MSLesSeg: median 0.89, 5th-95th percentile 0.77-1.20; its registration
# failures were at 1.6-1.7. Outside this range the two images most likely do not overlap -> check the case visually.
RATIO_RANGE = (0.7, 1.3)


def run(*command):
    subprocess.run([str(c) for c in command], check=True, env=FSL_ENV)


def check_fsl():
    missing = [tool for tool in ("flirt", "bet", "convert_xfm", "fslmaths") if shutil.which(tool) is None]
    if missing:
        raise RuntimeError(f"FSL tools not found on PATH: {missing}")


def brain_ml(path):
    image = sitk.ReadImage(path)
    return float((sitk.GetArrayFromImage(image) != 0).sum() * np.prod(image.GetSpacing()) / 1000)


def case_dirs(inputs):
    names = sorted(n for n in os.listdir(inputs) if os.path.isdir(os.path.join(inputs, n)))
    return [n for n in names if all(os.path.exists(os.path.join(inputs, n, f"{m.lower()}.nii.gz")) for m in cfg.modalities)]


def to_mni(inputs, out, template):
    if not os.path.exists(template):
        raise FileNotFoundError(f"MNI template {template} not found; pass --template "
                                f"(e.g. MSLesSeg-2024-main/resources/MNI152_T1_1mm.nii.gz)")
    cases = case_dirs(inputs)
    if not cases:
        raise FileNotFoundError(f"no <case>/flair.nii.gz + t1.nii.gz folders in {inputs}")
    qc_rows = []
    for case in cases:
        case_out = os.path.join(out, case)
        os.makedirs(case_out, exist_ok=True)
        for m in cfg.modalities:
            name = m.lower()
            registered = os.path.join(case_out, f"{name}.nii.gz")
            run("flirt", "-in", os.path.join(inputs, case, f"{name}.nii.gz"), "-ref", template,
                "-out", registered, "-omat", os.path.join(case_out, f"{name}_to_mni.mat"))
            run("bet", registered, registered)
        flair_ml, t1_ml = brain_ml(os.path.join(case_out, "flair.nii.gz")), brain_ml(os.path.join(case_out, "t1.nii.gz"))
        ratio = flair_ml / t1_ml if t1_ml else float("inf")
        flag = "" if RATIO_RANGE[0] <= ratio <= RATIO_RANGE[1] else "CHECK registration / brain extraction"
        qc_rows.append({"case": case, "flair_brain_ml": round(flair_ml, 1), "t1_brain_ml": round(t1_ml, 1),
                        "flair_t1_ratio": round(ratio, 3), "flag": flag})
        print(f"{case}: FLAIR/T1 brain volume ratio {ratio:.2f} {flag}")

    qc_file = os.path.join(out, "qc.csv")
    with open(qc_file, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(qc_rows[0]))
        writer.writeheader()
        writer.writerows(qc_rows)
    flagged = [r["case"] for r in qc_rows if r["flag"]]
    print(f"{len(qc_rows)} cases registered to {out}; flagged: {flagged or 'none'} (see {qc_file})")


def to_native(inputs, mni_dir, pred_dir, out):
    done = 0
    for case in case_dirs(inputs):
        prob = os.path.join(pred_dir, f"{case}_prob.nii.gz")
        forward = os.path.join(mni_dir, case, "flair_to_mni.mat")
        if not os.path.exists(prob) or not os.path.exists(forward):
            print(f"SKIP {case}: no prediction or no flair_to_mni.mat")
            continue
        case_out = os.path.join(out, case)
        os.makedirs(case_out, exist_ok=True)
        inverse = os.path.join(case_out, "mni_to_flair.mat")
        native_prob = os.path.join(case_out, "space-flair_prob-segmamba.nii.gz")
        run("convert_xfm", "-omat", inverse, "-inverse", forward)
        # resample the probability (not the binary mask) and threshold afterwards: gentler on thick FLAIR slices
        run("flirt", "-in", prob, "-ref", os.path.join(inputs, case, "flair.nii.gz"), "-applyxfm", "-init", inverse,
            "-interp", "trilinear", "-out", native_prob)
        run("fslmaths", native_prob, "-thr", cfg.threshold, "-bin", os.path.join(case_out, "space-flair_seg-segmamba.nii.gz"),
            "-odt", "char")
        done += 1
    print(f"{done} cases mapped to native FLAIR space in {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="mode", required=True)
    p = sub.add_parser("to_mni", help="register + skull-strip <inputs>/<case>/{flair,t1}.nii.gz")
    p.add_argument("--inputs", required=True, help="e.g. data/LSTAI_inputs (one folder per case)")
    p.add_argument("--out", required=True)
    p.add_argument("--template", default=os.path.join(os.environ.get("FSLDIR", ""), "data", "standard", "MNI152_T1_1mm.nii.gz"))
    p = sub.add_parser("to_native", help="map predict.py probabilities back onto <inputs>/<case>/flair.nii.gz")
    p.add_argument("--inputs", required=True)
    p.add_argument("--mni_dir", required=True, help="the --out folder of to_mni")
    p.add_argument("--pred_dir", required=True, help="the predict.py output folder for --input_dir <mni_dir>")
    p.add_argument("--out", required=True)
    args, _ = parser.parse_known_args()

    check_fsl()
    if args.mode == "to_mni":
        to_mni(cfg.user_path(args.inputs), cfg.user_path(args.out), cfg.user_path(args.template))
    else:
        to_native(cfg.user_path(args.inputs), cfg.user_path(args.mni_dir), cfg.user_path(args.pred_dir), cfg.user_path(args.out))
