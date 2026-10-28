"""Predict ventricles, normal WMH and MS lesions with a trained fold model (sliding window + 8x mirror TTA).

MS3SEG (official split):
    python segmamba_v2/predict.py --fold 1 --split val     -> prediction_results/segmamba_v2_ms3seg_fold1/val/
    python segmamba_v2/predict.py --fold 1 --split test    -> prediction_results/segmamba_v2_ms3seg_fold1/test/
Other data prepared like MS3SEG (T1 registered to FLAIR + a FLAIR brain mask, e.g. SVD after svd_flair.py), one
folder per case holding flair.nii.gz, t1.nii.gz and brain_mask.nii.gz:
    python segmamba_v2/predict.py --fold 1 --input_dir data/svd_flair  -> prediction_results/segmamba_v2_ms3seg_fold1/svd_flair/
Add --ckpt path/to/model.pt to choose a checkpoint (default: best_model_*.pt of the run).

Writes, on the input FLAIR grid with its full header:
  <case>.nii.gz               label map (uint8): 0 background, 1 ventricles, 2 normal WMH, 3 MS lesion
  <case>_prob_<class>.nii.gz  probability of each class (float32)
A voxel gets the most probable class whose probability exceeds 0.5. MS lesions smaller than the size in
logs/<run>/postprocessing.json (from evaluate.py --tune) are removed; without that file nothing is removed.
"""
import config as cfg  # must be first

import argparse
import glob
import json
import os

import numpy as np
import SimpleITK as sitk
import torch
from monai.inferers import SlidingWindowInferer
from monai.utils import set_determinism
from scipy import ndimage

from light_training.prediction import Predictor
from preprocess import FlairT1Preprocessor

set_determinism(123)


def find_checkpoint(ckpt):
    if ckpt is not None:
        return cfg.user_path(ckpt)
    candidates = glob.glob(os.path.join(cfg.model_save_path, "best_model_*.pt"))
    if not candidates:
        raise FileNotFoundError(f"no best_model_*.pt in {cfg.model_save_path}; pass --ckpt")
    return max(candidates, key=lambda p: float(os.path.basename(p)[len("best_model_"):-3]))


def load_state_dict(path):
    sd = torch.load(path, map_location="cpu")
    if "module" in sd:
        sd = sd["module"]
    return {(k[7:] if k.startswith("module") else k): v for k, v in sd.items()}


def ms3seg_cases(split):
    if not os.path.exists(cfg.cases_file):
        raise FileNotFoundError(f"{cfg.cases_file} not found; run segmamba_v2/preprocess.py first")
    cases = cfg.load_json(cfg.cases_file)
    return {c: {k: cases[c][k] for k in (*cfg.modalities, "BRAIN")} for c in cfg.split_cases(split)}


def folder_cases(input_dir):
    """{case: files} for every <input_dir>/<case>/ holding flair.nii.gz, t1.nii.gz and brain_mask.nii.gz."""
    cases = {}
    for name in sorted(os.listdir(input_dir)):
        if not os.path.isdir(os.path.join(input_dir, name)):
            continue
        files = {m: os.path.join(input_dir, name, f"{m.lower()}.nii.gz") for m in cfg.modalities}
        files["BRAIN"] = os.path.join(input_dir, name, "brain_mask.nii.gz")
        missing = [os.path.basename(p) for p in files.values() if not os.path.exists(p)]
        if missing:
            print(f"SKIP {name}: missing {missing}")
            continue
        cases[name] = files
    return cases


def remove_small_lesions(mask, min_size):
    """Removes connected components (26-connectivity) smaller than min_size voxels. Returns (mask, n_removed)."""
    if min_size <= 0 or not mask.any():
        return mask, 0
    labels, _ = ndimage.label(mask, structure=np.ones((3, 3, 3)))
    sizes = np.bincount(labels.ravel())
    small = np.flatnonzero(sizes < min_size)
    small = small[small > 0]
    return mask & ~np.isin(labels, small), len(small)


def to_label_map(prob):
    """(C, D, H, W) class probabilities -> label map: most probable class above the threshold, else 0."""
    labels = np.array(sorted(cfg.classes), dtype=np.uint8)[prob.argmax(axis=0)]
    return np.where(prob.max(axis=0) > cfg.threshold, labels, 0).astype(np.uint8)


def uncrop(array, properties):
    """Puts a (..., D, H, W) array of the brain crop back into the full input grid."""
    full = np.zeros((*array.shape[:-3], *properties["shape_before_cropping"]), dtype=array.dtype)
    (z0, z1), (y0, y1), (x0, x1) = properties["bbox_used_for_cropping"]
    full[..., z0:z1, y0:y1, x0:x1] = array
    return full


def save_like(array, reference, path):
    image = sitk.GetImageFromArray(array)
    image.CopyInformation(reference)  # origin, spacing and direction of the input FLAIR
    sitk.WriteImage(image, path)


def min_lesion_size():
    if not os.path.exists(cfg.postprocess_file):
        return 0
    return int(cfg.load_json(cfg.postprocess_file)["min_size"])


class ClassPredictor:
    def __init__(self, ckpt):
        print(f"loading {ckpt}")
        self.net = cfg.build_model()
        self.net.load_state_dict(load_state_dict(ckpt))
        self.net.eval()  # eval mode -> only the full-resolution output, no deep-supervision heads
        window_infer = SlidingWindowInferer(roi_size=cfg.roi_size,
                                            sw_batch_size=cfg.sw_batch_size,
                                            overlap=cfg.sw_overlap,
                                            progress=False,
                                            mode="gaussian")
        self.predictor = Predictor(window_infer=window_infer, mirror_axes=cfg.mirror_axes)

    def __call__(self, preprocessor, case_name, min_size):
        """Returns (label map, class probabilities, n small lesions removed) on the input FLAIR grid, sitk (z, y, x) order."""
        data, _, properties = preprocessor.run_case(case_name)  # skull-stripped, cropped, normalized, 1 mm
        logits = self.predictor.maybe_mirror_and_predict(torch.from_numpy(data)[None], self.net, device=cfg.device)
        logits = self.predictor.predict_raw_probability(logits, properties=properties)  # back to the native grid
        prob = torch.sigmoid(logits.float()).cpu().numpy()
        label_map = to_label_map(prob)
        lesion = label_map == cfg.lesion_label
        kept, n_removed = remove_small_lesions(lesion, min_size)
        label_map[lesion & ~kept] = 0
        prob[cfg.lesion_label - 1][lesion & ~kept] = 0  # removed lesions also leave the probability map
        torch.cuda.empty_cache()
        return uncrop(label_map, properties), uncrop(prob.astype(np.float32), properties), n_removed


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--split", choices=["val", "test"], help="MS3SEG validation split of --fold, or official test")
    source.add_argument("--input_dir", help="folder with <case>/flair.nii.gz, t1.nii.gz, brain_mask.nii.gz")
    parser.add_argument("--ckpt", type=str, default=None)
    parser.add_argument("--fold", type=int, default=1)  # parsed in config.py; selects the run folder
    parser.add_argument("--seed", type=int, default=123)  # parsed in config.py; selects the run folder
    args = parser.parse_args()

    if args.split:
        cases, save_dir = ms3seg_cases(args.split), cfg.pred_dir(args.split)
    else:
        input_dir = cfg.user_path(args.input_dir)
        cases, save_dir = folder_cases(input_dir), cfg.pred_dir(os.path.basename(os.path.normpath(input_dir)))
    if not cases:
        raise FileNotFoundError("no cases to predict")

    ckpt = find_checkpoint(args.ckpt)
    min_size = min_lesion_size()
    print(f"{len(cases)} cases -> {save_dir}; removing predicted MS lesions < {min_size} voxels")
    os.makedirs(save_dir, exist_ok=True)
    with open(os.path.join(save_dir, "prediction_info.json"), "w") as f:
        json.dump({"checkpoint": ckpt, "fold": cfg.fold, "min_size": min_size, "classes": cfg.classes,
                   "cases": list(cases)}, f, indent=1)

    class_predictor = ClassPredictor(ckpt)
    preprocessor = FlairT1Preprocessor(cases)
    for i, (case_name, files) in enumerate(cases.items()):
        label_map, prob, n_removed = class_predictor(preprocessor, case_name, min_size)
        reference = sitk.ReadImage(files[cfg.modalities[0]])
        save_like(label_map, reference, os.path.join(save_dir, f"{case_name}.nii.gz"))
        for label, name in cfg.classes.items():
            save_like(prob[label - 1], reference, os.path.join(save_dir, f"{case_name}_prob_{name}.nii.gz"))
        voxel_ml = np.prod(reference.GetSpacing()) / 1000
        volumes = ", ".join(f"{name} {(label_map == label).sum() * voxel_ml:.1f} ml" for label, name in cfg.classes.items())
        print(f"[{i + 1}/{len(cases)}] {case_name}: {volumes} ({n_removed} small lesions removed)")
    print(f"saved to {save_dir}")
