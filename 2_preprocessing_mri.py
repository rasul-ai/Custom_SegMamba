
import os

import numpy as np
import SimpleITK as sitk

import brats2023
from light_training.preprocessing.preprocessors.preprocessor_mri import MultiModalityPreprocessor

seg_filename = "seg"

base_dir = os.path.dirname(brats2023.raw_data_dir)
image_dir = os.path.basename(brats2023.raw_data_dir)


class BraTS2023Preprocessor(MultiModalityPreprocessor):
    """Reads <case>/<case>-<modality>.nii.gz directly, so the raw BraTS2023 files need no renaming."""

    def read_data(self, case_name):
        assert len(self.data_filenames) != 0

        data = []
        spacing = None
        for modality in self.data_filenames:
            image = sitk.ReadImage(brats2023.case_file(case_name, modality))
            spacing = image.GetSpacing()
            data.append(sitk.GetArrayFromImage(image).astype(np.float32)[None])

        data = np.concatenate(data, axis=0)

        seg_arr = None
        if self.seg_filename != "":
            seg = sitk.ReadImage(brats2023.case_file(case_name, self.seg_filename))
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
    preprocessor = BraTS2023Preprocessor(
        base_dir=base_dir,
        image_dir=image_dir,
        data_filenames=brats2023.modalities,
        seg_filename=seg_filename,
    )

    out_spacing = [1.0, 1.0, 1.0]
    output_dir = brats2023.data_dir

    preprocessor.run(output_spacing=out_spacing,
                     output_dir=output_dir,
                     all_labels=[1, 2, 3],
    )

def plan():
    preprocessor = BraTS2023Preprocessor(
        base_dir=base_dir,
        image_dir=image_dir,
        data_filenames=brats2023.modalities,
        seg_filename=seg_filename,
    )

    preprocessor.run_plan()


if __name__ == "__main__":

    plan()
    process_train()

