# SegMamba for MS lesion segmentation (branch `msles2025`)

This branch trains SegMamba v2 on the **MSLesSeg** dataset (FLAIR + T1 → MS lesion mask). It then tests the model on other FLAIR + T1 data, such as the lab's SVD cohort, where LST-AI provides the reference masks. Branch `main` keeps the BraTS2020 experiments (SegMamba, SegMamba v2, Swin UNETR).

**Workflow (preprocess, train, predict, evaluate, SVD test): [segmamba_v2/README.md](segmamba_v2/README.md)**

SegMamba: Long-range Sequential Modeling Mamba For 3D Medical Image Segmentation, [https://arxiv.org/abs/2401.13560](https://arxiv.org/abs/2401.13560), original code at [https://github.com/ge-xing/SegMamba](https://github.com/ge-xing/SegMamba).

## Repository layout

| Folder | Contents |
|---|---|
| `segmamba_v2/` | The MSLesSeg scripts: `config.py`, `preprocess.py`, `train.py`, `predict.py`, `evaluate.py`, `svd_mni.py`, `check_gpu.py` |
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

`segmamba_v2/svd_mni.py` also needs [FSL](https://fsl.fmrib.ox.ac.uk) (`flirt`, `bet`, `convert_xfm`, `fslmaths`), but only on the machine that registers the SVD scans.

## Acknowledgement

Many thanks to these repos for their contributions:

[https://github.com/MIC-DKFZ/nnUNet](https://github.com/MIC-DKFZ/nnUNet)

[https://github.com/Project-MONAI/MONAI](https://github.com/Project-MONAI/MONAI)

[https://github.com/hustvl/Vim](https://github.com/hustvl/Vim)

[https://github.com/bowang-lab/U-Mamba](https://github.com/bowang-lab/U-Mamba)
