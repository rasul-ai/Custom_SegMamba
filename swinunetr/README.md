# Swin UNETR baseline (BraTS2020), for comparison with SegMamba

This folder trains and evaluates [Swin UNETR](https://arxiv.org/abs/2201.01266) using the same pipeline as SegMamba. The comparison stays controlled because both models share:

| | SegMamba (`3_train.py`) | Swin UNETR (this folder) |
|---|---|---|
| Data | `data/fullres/train` | same |
| Split | `get_train_val_test_loader_from_train` (70/10/20, seed 42) | same |
| Patch / batch | 128³ / 2 | same |
| Augmentation | `light_training` nnU-Net-style | same |
| Budget | 1000 epochs × 250 steps, validation every 2 epochs | same |
| Loss | 4-class cross entropy | same (`loss_name = "ce"`) |
| Optimizer | SGD lr 1e-2, poly | AdamW lr 1e-4, warmup + cosine (`optimizer_name = "sgd"` for SegMamba's) |
| Inference | sliding window 0.5 overlap, gaussian, 8× mirror TTA | same |
| Metrics | `5_compute_metrics.py` (Dice + HD95 for TC/WT/ET) | same script |

All settings are in `config.py`.

## Usage (run on the GPU machine, from the repo root)

```bash
# 0. optional: check peak GPU memory for the configured batch/patch size (takes about a minute)
python swinunetr/check_gpu_memory.py

# 1. train  -> logs/swinunetr/ (tensorboard + checkpoints + data_split.json)
python swinunetr/train.py

# 2. predict the test split -> prediction_results/swinunetr/
python swinunetr/predict.py --ckpt logs/swinunetr/model/best_model_0.7493.pt

# 3. metrics, using the same script as SegMamba
python 5_compute_metrics.py --pred_name swinunetr
python 5_compute_metrics.py --pred_name segmamba

# 4. side-by-side table with a paired Wilcoxon test
python swinunetr/compare_models.py
```

## GPU memory (RTX 4090, 24 GB)

Swin UNETR at 128³ with feature_size 48 is memory-hungry. Without gradient checkpointing, batch size 2 is likely to exceed 24 GB. `config.py` therefore sets
`use_checkpoint = True`, which recomputes activations in the backward pass and costs roughly 20–30 % more time per step. It also uses AMP (from the shared trainer) and
`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`. `train.py` prints the peak memory after every validation.
If `check_gpu_memory.py` still reports that memory is tight, set `batch_size = 1`. For inference, lower `sw_batch_size` to 1.

## Notes

- **The split must match SegMamba's.** The split shuffles `glob()` output, and file order can differ between machines. Train on the same machine and with the same
  `data/fullres/train` folder that SegMamba used. `train.py` saves the split to `logs/swinunetr/data_split.json`, and `predict.py` refuses to run if the test set differs.
- Swin UNETR is trained from scratch, like SegMamba. No pretrained weights are used.
