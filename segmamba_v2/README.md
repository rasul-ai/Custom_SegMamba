# SegMamba v2: Phase A (fixed training recipe)

Same network backbone (`model_segmamba/segmamba.py`, unchanged), data, split, patch size, augmentation and inference as `3_train.py` / `4_predict.py`. Only the training recipe changes. Nothing outside this folder is modified, and all outputs go to new paths (`logs/segmamba_v2*`, `prediction_results/segmamba_v2*`).

| | Original SegMamba (`3_train.py`) | SegMamba v2 (this folder) |
|---|---|---|
| LR schedule | SGD 1e-2, poly over **1000** epochs, stopped at 266 → LR still ≈ 7.6e-3 | SGD 1e-2, poly over **250** epochs → LR decays to 0 |
| Output | 4-class softmax | 3 sigmoid regions (TC, WT, ET) |
| Loss | cross entropy | BCE + soft Dice (batch Dice) |
| Deep supervision | no | yes: extra 1×1×1 heads at 1/2, 1/4, 1/8 (weights 1, ½, ¼, ⅛, normalized) |
| Region consistency | from argmax | thresholds at 0.5, then ET ⊂ TC ⊂ WT enforced (nnU-Net region order) |
| Post-processing | none | ET small-component removal + ET→necrosis below a volume threshold, tuned on validation |
| Logs | tensorboard only | tensorboard + `train_log.csv` (mean loss per epoch) + `val_log.csv` |
| Split | `get_train_val_test_loader_from_train` (70/10/20, seed 42) | same function; `train.py` also checks it against `logs/swinunetr/data_split.json` and stops if it differs |
| Seeds | 123 (init), then 42 (training) | same by default; `--seed N` shifts both for repeat runs → `logs/segmamba_v2_seedN/` |

Settings are in `config.py`. Set `optimizer_name = "adamw"` to use the Swin UNETR optimizer recipe instead.
Deep supervision adds only 1×1×1 convolutions, so memory is essentially the same as the original SegMamba. `train.py` prints the peak GPU memory after each validation.

## Usage (on the GPU machine, from the repo root)

```bash
# 1. train (250 epochs x 250 steps, validation every 2 epochs)
python segmamba_v2/train.py                 # optional: --seed 7, --seed 2024 for repeat runs

# 2. predict validation + test (no post-processing)
python segmamba_v2/predict.py --split val   # -> prediction_results/segmamba_v2_val/
python segmamba_v2/predict.py --split test  # -> prediction_results/segmamba_v2/

# 3. tune ET post-processing on validation, then apply it to test
python segmamba_v2/postprocess.py tune      # -> logs/segmamba_v2/postprocessing.json
python segmamba_v2/postprocess.py apply     # -> prediction_results/segmamba_v2_pp/

# 4. metrics (both conventions, see below)
python segmamba_v2/evaluate.py --pred_name segmamba_v2    --split test
python segmamba_v2/evaluate.py --pred_name segmamba_v2_pp --split test

# 5. compare with the Sep 30 models (legacy convention, paired Wilcoxon test)
python swinunetr/compare_models.py --models segmamba_v2_pp swinunetr
python swinunetr/compare_models.py --models segmamba_v2_pp segmamba
```

For another seed, pass the same `--seed N` to every command.

## Metric conventions

`5_compute_metrics.py`, and therefore the Sep 30 report, scores a region as Dice 0 / HD95 50 whenever the prediction **or** the ground truth is empty. That includes a correct empty prediction for a case without enhancing tumor. Under that convention, removing false-positive ET can never raise the score on those cases. `evaluate.py` therefore reports both:

- **legacy**: identical to `5_compute_metrics.py`. Use it to compare with the Sep 30 numbers. It's saved as `<pred_name>.npy`, so `compare_models.py` works.
- **empty-aware**: if both are empty, Dice is 1 and HD95 is 0. This is closer to the BraTS evaluation and is what post-processing is tuned for. It's saved as `<pred_name>_empty_aware.npy`.
