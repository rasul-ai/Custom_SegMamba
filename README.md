# SegMamba for MS3SEG: ventricles, normal WMH and MS lesions (branch `ms3seg2026`)

This branch trains SegMamba v2 on the **MS3SEG** dataset (FLAIR + T1 → ventricles, normal white-matter hyperintensities, MS lesions). It then tests the model on other FLAIR + T1 data, such as the lab's SVD cohort, where LST-AI provides the reference masks. Related branches:
- `msles2025`: the same pipeline on MSLesSeg (binary MS lesions, MNI space).
- `main`: the BraTS2020 experiments.

**Workflow (preprocess, train, predict, evaluate, SVD test): [segmamba_v2/README.md](segmamba_v2/README.md)**

SegMamba: Long-range Sequential Modeling Mamba For 3D Medical Image Segmentation, [https://arxiv.org/abs/2401.13560](https://arxiv.org/abs/2401.13560), original code at [https://github.com/ge-xing/SegMamba](https://github.com/ge-xing/SegMamba). MS3SEG dataset and official split: [https://github.com/Mahdi-Bashiri/MS3SEG](https://github.com/Mahdi-Bashiri/MS3SEG).

## Repository layout

| Folder | Contents |
|---|---|
| `segmamba_v2/` | The MS3SEG scripts: `config.py`, `preprocess.py`, `train.py`, `predict.py`, `evaluate.py`, `svd_flair.py`, `check_gpu.py`, plus the official split `ms3seg_splits.json` |
| `model_segmamba/` | The SegMamba network (unchanged) |
| `light_training/` | Data loading, augmentation, trainer and sliding-window predictor |
| `monai/` | Vendored MONAI, which the scripts import ahead of any installed copy |
| `mamba/`, `causal-conv1d/` | CUDA extensions that SegMamba needs; build them as below |

## Environment install (GPU machine)

```bash
pip install -U pip setuptools wheel
pip install -r requirements.txt     # change cu118 in the file to match your CUDA version

cd causal-conv1d && python setup.py install && cd ..
cd mamba && python setup.py install && cd ..

python segmamba_v2/check_gpu.py     # checks the CUDA build and the GPU memory needed for training
```

`segmamba_v2/svd_flair.py` also needs [FSL](https://fsl.fmrib.ox.ac.uk) (`flirt`, `bet`), but only on the machine that prepares the SVD scans.

## Acknowledgement

Many thanks to these repos for their contributions:

[https://github.com/MIC-DKFZ/nnUNet](https://github.com/MIC-DKFZ/nnUNet)

[https://github.com/Project-MONAI/MONAI](https://github.com/Project-MONAI/MONAI)

[https://github.com/hustvl/Vim](https://github.com/hustvl/Vim)

[https://github.com/bowang-lab/U-Mamba](https://github.com/bowang-lab/U-Mamba)

[https://github.com/Mahdi-Bashiri/MS3SEG](https://github.com/Mahdi-Bashiri/MS3SEG)
