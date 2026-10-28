# SegMamba

**Recent news: If you are interested in the research about vision language models, please refers to the latest work: https://github.com/MrGiovanni/RadGPT (ICCV2025)**

**Now we have open-sourced the pre-processing, training, inference, and metrics computation codes.**

> **Branch `brats2023`:** this branch trains SegMamba (original recipe and v2) and the Swin UNETR baseline on BraTS2023 (GLI); see [the BraTS2023 section](#brats2023-preprocessing-training-testing-inference-and-metrics-computation-branch-brats2023). Branch `main` holds the BraTS2020 runs.

SegMamba: Long-range Sequential Modeling Mamba For 3D Medical Image Segmentation

[https://arxiv.org/abs/2401.13560](https://arxiv.org/abs/2401.13560)

![](images/method_figure.jpg)

![](images/modules.jpg)

Our advantage in speed and memory.
![](images/segmamba_ablation.jpg)

## Contact 
If you have any questions about our project, please feel free to contact us by email at zxing565@connect.hkust-gz.edu.cn or via WeChat at 18340097191. Furthermore, the data underlying this article will be shared on reasonable request to gaof57@mail.sysu.edu.cn.

## Environment install
Clone this repository and navigate to the root directory of the project.

```bash
git clone https://github.com/ge-xing/SegMamba.git

cd SegMamba
```
### Install causal-conv1d

```bash
cd causal-conv1d

python setup.py install
```

### Install mamba

```bash
cd mamba

python setup.py install
```

### Install monai 

```bash
pip install monai
```

## Simple test

```bash
python 0_inference.py
```

## BraTS2023: preprocessing, training, testing, inference, and metrics computation (branch `brats2023`)

This branch trains on the BraTS2023 glioma (GLI) training data. Branch `main` keeps the BraTS2020 runs. All BraTS2023 settings are in `brats2023.py`:
- raw-data path and file names;
- channel order: FLAIR, T2, T1, T1ce, i.e. `t2f`, `t2w`, `t1n`, `t1c`;
- the patient-level split;
- output paths. Preprocessed data, logs and predictions go to their own `brats2023` folders, so they never mix with the BraTS2020 ones in the same checkout.

### Data

Put the extracted training data at `data/BraTS2023/ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData/`, or point `BRATS2023_DIR` to it. Each case is `<case>/<case>-{t2f,t2w,t1n,t1c,seg}.nii.gz` and is read as it is; there is no renaming step any more.

- 1,251 cases, all 240×240×155 at 1 mm.
- Labels are 1 (necrotic core), 2 (edema), 3 (enhancing tumor). BraTS2020 used 4 for ET.
- The challenge's `ValidationData` has no segmentations, so it is not used.

### Preprocessing

Resampling, normalization and cropping, as before:

```bash
python 2_preprocessing_mri.py
```

- Output goes to `data/fullres_brats2023/train/`; set `BRATS2023_FULLRES_DIR` to put it elsewhere.
- Plan for about **90 GB** of disk: the `.npz` files take 18 GB, and training unpacks them to `.npy` (70 GB) next to them.
- The first step (`plan()`) only analyses the data and writes `data_analysis_result.txt`.

### Split

`brats2023.get_train_val_test_loader` makes a 70/10/20 train / validation / test split **by patient**, with seed 42.
- 118 patients have two scans (`...-000` and a follow-up `...-001`). Both scans of a patient always land in the same split, so no patient is in both training and test.
- Patients are sorted before the seeded shuffle, so the split does not depend on the file-listing order.
- All models (`3_train.py`, `segmamba_v2/`, `swinunetr/`) use this split.

### Training

```bash
python 3_train.py                  # original SegMamba recipe -> logs/brats2023/segmamba/
python segmamba_v2/train.py        # SegMamba v2 recipe      -> logs/brats2023/segmamba_v2/  (see segmamba_v2/README.md)
python swinunetr/train.py          # Swin UNETR baseline     -> logs/brats2023/swinunetr/    (see swinunetr/README.md)
```

### Inference

```bash
python 4_predict.py                # best checkpoint of 3_train.py -> prediction_results/brats2023/segmamba/
```

`segmamba_v2/predict.py` and `swinunetr/predict.py` write to `prediction_results/brats2023/<model>/`.

### Metrics computation

Dice and HD95 for each region (WT, TC, ET) on the test split:

```bash
python 5_compute_metrics.py --pred_name segmamba      # -> prediction_results/brats2023/result_metrics/segmamba.npy
python swinunetr/compare_models.py --models segmamba swinunetr
```

### Analyze Tensorboard files
```bash
python -m tensorboard.main --logdir logs/brats2023
```



## Acknowledgement
Many thanks for these repos for their great contribution!

[https://github.com/MIC-DKFZ/nnUNet](https://github.com/MIC-DKFZ/nnUNet)

[https://github.com/Project-MONAI/MONAI](https://github.com/Project-MONAI/MONAI)

[https://github.com/hustvl/Vim](https://github.com/hustvl/Vim)

[https://github.com/bowang-lab/U-Mamba](https://github.com/bowang-lab/U-Mamba)

