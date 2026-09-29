
import os

import numpy as np
import SimpleITK as sitk

from light_training.preprocessing.preprocessors.preprocessor_mri import MultiModalityPreprocessor

data_filename = ["flair.nii", "t2.nii", "t1.nii", "t1ce.nii"]
seg_filename = "seg.nii"

base_dir = "data/BraTS2020_TrainingData"
image_dir = "MICCAI_BraTS2020_TrainingData"


class BraTS2020Preprocessor(MultiModalityPreprocessor):
    def _resolve_case_file(self, case_name: str, filename: str) -> str:
        case_dir = os.path.join(self.base_dir, self.image_dir, case_name)
        direct_path = os.path.join(case_dir, filename)
        if os.path.exists(direct_path):
            return direct_path

        # Raw BraTS2020 files are often stored as: <case_name>_<modality>.nii
        prefixed_path = os.path.join(case_dir, f"{case_name}_{filename}")
        if os.path.exists(prefixed_path):
            return prefixed_path

        raise FileNotFoundError(
            f"Cannot find '{filename}' or '{case_name}_{filename}' in '{case_dir}'"
        )

    def read_data(self, case_name):
        assert len(self.data_filenames) != 0

        data = []
        spacing = None
        for dfname in self.data_filenames:
            image_path = self._resolve_case_file(case_name, dfname)
            image = sitk.ReadImage(image_path)
            spacing = image.GetSpacing()
            data.append(sitk.GetArrayFromImage(image).astype(np.float32)[None])

        data = np.concatenate(data, axis=0)

        seg_arr = None
        if self.seg_filename != "":
            seg_path = self._resolve_case_file(case_name, self.seg_filename)
            seg = sitk.ReadImage(seg_path)
            seg_arr = sitk.GetArrayFromImage(seg).astype(np.float32)[None]
            intensities_per_channel, intensity_statistics_per_channel = (
                self.collect_foreground_intensities(seg_arr, data)
            )
        else:
            intensities_per_channel = []
            intensity_statistics_per_channel = []

        properties = {
            "spacing": spacing,
            "raw_size": data.shape[1:],
            "name": case_name.split(".")[0],
            "intensities_per_channel": intensities_per_channel,
            "intensity_statistics_per_channel": intensity_statistics_per_channel,
        }

        return data, seg_arr, properties

def process_train():
    preprocessor = BraTS2020Preprocessor(
        base_dir=base_dir,
        image_dir=image_dir,
        data_filenames=data_filename,
        seg_filename=seg_filename,
    )

    out_spacing = [1.0, 1.0, 1.0]
    output_dir = "data/fullres/train/"
    
    preprocessor.run(output_spacing=out_spacing, 
                     output_dir=output_dir, 
                     all_labels=[1, 2, 3],
    )

def plan():
    preprocessor = BraTS2020Preprocessor(
        base_dir=base_dir,
        image_dir=image_dir,
        data_filenames=data_filename,
        seg_filename=seg_filename,
    )
    
    preprocessor.run_plan()


if __name__ == "__main__":

    plan()
    process_train()

