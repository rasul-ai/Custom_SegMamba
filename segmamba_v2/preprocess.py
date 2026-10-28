"""Preprocess MS3SEG for training.

Run:  python segmamba_v2/preprocess.py --ms3seg_dir "/path/to/MS3SEG_2026"
      (the folder holding the extracted MS_100_patient_registered/ and MS_100_patient_masks/)

Writes to data/ms3seg_fullres/:
  cases.json        every patient: FLAIR / T1 / brain mask / class mask paths, class volumes, official split
  <id>.npz/.pkl     cropped, normalized FLAIR + T1 (1 mm slices) and the label map of the 80 fold patients (for train.py)
The folds and the test set are the official MS3SEG ones (segmamba_v2/ms3seg_splits.json).

MS3SEG is in native FLAIR space: T1 registered to FLAIR (FSL flirt) and a FLAIR brain mask (FSL bet), both
provided. Preprocessing: skull-strip with that brain mask, crop to the brain, z-score each channel inside the brain,
resample to config.out_spacing (slices to 1 mm, in-plane native), record foreground locations for patch sampling.
predict.py runs the same FlairT1Preprocessor
on every case it predicts.
"""
import config as cfg  # must be first: sets sys.path, cwd and CUDA allocator config

import argparse
import glob
import json
import os

import numpy as np
import SimpleITK as sitk

from light_training.preprocessing.preprocessors.preprocessor_mri import MultiModalityPreprocessor


class FlairT1Preprocessor(MultiModalityPreprocessor):
    """cases: {case_name: {"FLAIR": path, "T1": path, "BRAIN": brain mask path, optional "MASKS": {label: path}}}."""

    def __init__(self, cases):
        super().__init__(base_dir="", image_dir="", data_filenames=cfg.modalities, seg_filename="")
        self.cases = cases
        # run_case_npy() needs these even when run() is not called (prediction)
        self.out_spacing = cfg.out_spacing
        self.all_labels = list(cfg.classes)
        self.foreground_intensity_properties_per_channel = {}

    def get_iterable_list(self):
        return list(self.cases)

    def _normalize(self, data, seg, foreground_intensity_properties_per_channel):
        # z-score inside the brain (nonzero after read_data), 0 outside
        for c in range(data.shape[0]):
            brain = data[c] != 0
            if brain.any():
                values = data[c][brain]
                data[c] = np.where(brain, (data[c] - values.mean()) / max(values.std(), 1e-8), 0)
        return data

    def read_data(self, case_name):
        files = self.cases[case_name]
        images = [sitk.ReadImage(files[m]) for m in cfg.modalities]
        brain_image = sitk.ReadImage(files["BRAIN"])
        ref = images[0]
        for name, image in list(zip(cfg.modalities[1:], images[1:])) + [("brain mask", brain_image)]:
            if image.GetSize() != ref.GetSize() or not np.allclose(image.GetSpacing(), ref.GetSpacing(), atol=1e-3):
                raise ValueError(f"{case_name}: {name} grid {image.GetSize()} {image.GetSpacing()} differs from "
                                 f"{cfg.modalities[0]} {ref.GetSize()} {ref.GetSpacing()}; register it to FLAIR first")
        brain = sitk.GetArrayFromImage(brain_image) > 0
        data = np.stack([sitk.GetArrayFromImage(image).astype(np.float32) for image in images])
        # Skull-strip with the brain mask: 0 outside, and inside shifted so the minimum is 1. Cropping and
        # _normalize treat nonzero voxels as brain, and raw FLAIR is <= 0 in parts of the ventricles.
        # The z-score in _normalize removes the shift again.
        for c in range(data.shape[0]):
            data[c] = np.where(brain, data[c] - data[c][brain].min() + 1, 0)

        seg = None
        if files.get("MASKS"):
            # label map; the classes barely overlap, where they do the higher label (MS lesion) wins
            seg = np.zeros(data.shape[1:], dtype=np.float32)
            for label in sorted(cfg.classes):
                seg[sitk.GetArrayFromImage(sitk.ReadImage(files["MASKS"][str(label)])) > 0] = label
            seg = seg[None]

        properties = {"spacing": ref.GetSpacing(),
                      "raw_size": data.shape[1:],
                      "name": case_name}
        return data, seg, properties


def find_cases(ms3seg_dir):
    """{patient id: files} from MS_100_patient_registered/<id>/ and MS_100_patient_masks/<class>_Masks/<id>/."""
    registered = os.path.join(ms3seg_dir, "MS_100_patient_registered")
    masks = os.path.join(ms3seg_dir, "MS_100_patient_masks")
    cases = {}
    for case_dir in sorted(glob.glob(os.path.join(registered, "*"))):
        case = os.path.basename(case_dir)
        info = {"FLAIR": os.path.join(case_dir, f"{case}_FLAIR.nii.gz"),
                "T1": os.path.join(case_dir, f"{case}_T1WI_reg.nii.gz"),
                "BRAIN": os.path.join(case_dir, f"{case}_brain_mask.nii.gz"),
                "MASKS": {str(label): os.path.join(masks, f"{folder}_Masks", case, f"{case}_{folder}_Mask.nii.gz")
                          for label, folder in cfg.mask_folders.items()}}
        missing = [p for p in [info["FLAIR"], info["T1"], info["BRAIN"], *info["MASKS"].values()] if not os.path.exists(p)]
        if missing:
            print(f"SKIP {case}: missing {[os.path.basename(p) for p in missing]}")
            continue
        cases[case] = info
    return cases


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ms3seg_dir", required=True,
                        help="folder holding MS_100_patient_registered/ and MS_100_patient_masks/")
    parser.add_argument("--num_processes", type=int, default=8)
    args, _ = parser.parse_known_args()

    cases = find_cases(cfg.user_path(args.ms3seg_dir))
    if not cases:
        raise FileNotFoundError(f"no patients under {args.ms3seg_dir}/MS_100_patient_registered")
    test = set(cfg.split_cases("test"))
    for case, info in cases.items():
        info["split"] = "test" if case in test else "train"
        for label, path in info["MASKS"].items():
            mask = sitk.ReadImage(path)
            ml = float((sitk.GetArrayFromImage(mask) > 0).sum() * np.prod(mask.GetSpacing()) / 1000)
            info[f"{cfg.classes[int(label)]}_ml"] = ml

    official = set(cfg.split_cases("train")) | set(cfg.split_cases("val")) | test
    if official - set(cases):
        raise FileNotFoundError(f"patients of the official split missing in {args.ms3seg_dir}: {sorted(official - set(cases))}")

    os.makedirs(cfg.data_dir, exist_ok=True)
    with open(cfg.cases_file, "w") as f:
        json.dump(cases, f, indent=1)

    fold_cases = sorted(set(cfg.split_cases("train")) | set(cfg.split_cases("val")))
    print(f"{len(cases)} patients: {len(fold_cases)} in the folds, {len(test)} official test")
    for label, name in cfg.classes.items():
        v = [cases[c][f"{name}_ml"] for c in cases]
        print(f"  {name}: median {np.median(v):.1f} ml (range {min(v):.1f}-{max(v):.1f})")

    # Unpacked .npy files of an earlier run would be reused as they are (dataset.py never overwrites them).
    for stale in glob.glob(os.path.join(cfg.data_dir, "*.npy")):
        os.remove(stale)

    preprocessor = FlairT1Preprocessor({c: cases[c] for c in fold_cases})
    preprocessor.run(output_spacing=cfg.out_spacing,
                     output_dir=cfg.data_dir,
                     all_labels=list(cfg.classes),
                     num_processes=args.num_processes)
    print(f"saved {len(fold_cases)} cases to {cfg.data_dir}")
