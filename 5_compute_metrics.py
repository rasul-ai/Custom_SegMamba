import brats2023
from brats2023 import get_train_val_test_loader
from monai.utils import set_determinism
import torch 
import os
import numpy as np
import SimpleITK as sitk
from medpy import metric
import argparse
from tqdm import tqdm 

import numpy as np

set_determinism(123)

parser = argparse.ArgumentParser()

parser.add_argument("--pred_name", required=True, type=str)

results_root = brats2023.results_root
args = parser.parse_args()

pred_name = args.pred_name


def resolve_existing_path(candidates):
    for path in candidates:
        if os.path.exists(path):
            return path
    raise FileNotFoundError(f"None of these paths exists: {candidates}")

def cal_metric(gt, pred, voxel_spacing):
    if pred.sum() > 0 and gt.sum() > 0:
        dice = metric.binary.dc(pred, gt)
        hd95 = metric.binary.hd95(pred, gt, voxelspacing=voxel_spacing)
        return np.array([dice, hd95])
    else:
        return np.array([0.0, 50])

def each_cases_metric(gt, pred, voxel_spacing):
    classes_num = 3
    class_wise_metric = np.zeros((classes_num, 2))
    for cls in range(0, classes_num):
        class_wise_metric[cls, ...] = cal_metric(pred[cls], gt[cls], voxel_spacing)
    print(class_wise_metric)
    return class_wise_metric

def convert_labels(labels):
    ## TC, WT and ET
    labels = labels.unsqueeze(dim=0)

    result = [(labels == 1) | (labels == 3), (labels == 1) | (labels == 3) | (labels == 2), labels == 3]
    
    return torch.cat(result, dim=0).float()


if __name__ == "__main__":
    data_dir = brats2023.data_dir
    train_ds, val_ds, test_ds = get_train_val_test_loader(data_dir)
    print(len(test_ds))
    all_results = np.zeros((len(test_ds), 3, 2), dtype=np.float32)

    ind = 0
    for batch in tqdm(test_ds, total=len(test_ds)):
        properties = batch["properties"]
        case_name = properties["name"]

        gt_itk = brats2023.case_file(case_name, "seg")
        voxel_spacing = [1, 1, 1]
        gt_itk = sitk.ReadImage(gt_itk)
        gt_array = sitk.GetArrayFromImage(gt_itk).astype(np.int32)
        gt_array = torch.from_numpy(gt_array)
        gt_array = convert_labels(gt_array).numpy()

        pred_itk_path = resolve_existing_path([
            os.path.join(results_root, pred_name, f"{case_name}.nii.gz"),
            os.path.join(results_root, pred_name, f"{case_name}.nii"),
        ])
        pred_itk = sitk.ReadImage(pred_itk_path)
        pred_array = sitk.GetArrayFromImage(pred_itk)

        m = each_cases_metric(gt_array, pred_array, voxel_spacing)

        all_results[ind, ...] = m
    
        ind += 1

    os.makedirs(os.path.join(results_root, "result_metrics"), exist_ok=True)
    np.save(os.path.join(results_root, "result_metrics", f"{pred_name}.npy"), all_results) 
    
    result = np.load(os.path.join(results_root, "result_metrics", f"{pred_name}.npy"))
    print(result.shape)
    print(result.mean(axis=0))
    print(result.std(axis=0))



