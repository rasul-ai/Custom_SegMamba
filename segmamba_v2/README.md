# SegMamba v2 on MS3SEG (FLAIR + T1 → ventricles, normal WMH, MS lesions)

Same network and training recipe as SegMamba v2 on BraTS (branch `main`) and on MSLesSeg (branch `msles2025`). The changes are for MS3SEG's data and labels:

| | `msles2025` (MSLesSeg) | This branch (MS3SEG) |
|---|---|---|
| Input | FLAIR + T1 | FLAIR + T1 (MS3SEG's T1 registered to FLAIR) |
| Output | lesion (1 sigmoid channel) | **ventricles, normal WMH, MS lesion** (3 sigmoid channels), saved as a label map 0–3 with MS3SEG's values |
| Space | MNI152 1 mm | **native FLAIR space**: 2D axial, 0.9 mm in-plane, 6–8 mm slices. Slices are resampled to 1 mm for the network and predictions go back onto the native grid |
| Skull stripping | FSL `bet` (already applied) | the dataset's `brain_mask.nii.gz` (FSL `bet -f 0.5`) |
| Split | own patient-level 5 folds | **official MS3SEG split**: folds 1–5 (64 train / 16 val patients) + 20 held-out test patients (`ms3seg_splits.json`, from the MS3SEG repository) |
| Excluded scans | 7 QC failures | none; QC found no broken or misaligned patients |
| Unchanged | | SGD 1e-2 + poly over 250 epochs, BCE + soft Dice, deep supervision, 128³ patches, batch 2, augmentation, sliding window + 8× mirror TTA, z-score inside the brain, minimum MS lesion size tuned on validation |

All settings are in `config.py`.

## 0. Set up the GPU PC (once)

1. Check out branch `ms3seg2026` and install the environment: see the top-level README.
2. Copy the extracted `MS_100_patient_registered/` and `MS_100_patient_masks/` folders there, inside one folder such as `MS3SEG_2026/`. They come from `MS_100_patient_registered.rar` (600 MB) and `MS_100_patient_masks.rar` (4 MB); the other archives are not needed.
3. Check that the CUDA extensions work and that training fits in memory (about a minute):
   ```bash
   python segmamba_v2/check_gpu.py
   ```

## 1. Preprocess MS3SEG (about 1 minute)

```bash
python segmamba_v2/preprocess.py --ms3seg_dir /path/to/MS3SEG_2026
```
This writes `data/ms3seg_fullres/` (set `MS3SEG_DATA_DIR` to put it elsewhere):
- `cases.json`: every patient with its files, class volumes and split.
- `<id>.npz/.pkl`: the 80 fold patients, skull-stripped, cropped, normalized, with 1 mm slices.

Plan for about 5 GB of disk: the `.npz` files take 2.1 GB, and training unpacks them to `.npy` (2.5 GB) next to them.

## 2. Train (about 10 h per fold on an RTX 4090)

```bash
python segmamba_v2/train.py --fold 1
tensorboard --logdir logs          # optional
```
- Output goes to `logs/segmamba_v2_ms3seg_fold1/`: `train_log.csv`, `val_log.csv` (patch Dice per class and their mean), `data_split.json` and `model/best_model_<mean dice>.pt`.
- Folds are numbered 1–5, like the official split and the MS3SEG paper's tables. Start with fold 1; folds 2–5 give the cross-validated estimate the paper reports.
- `--seed N` repeats a fold with another seed. Pass the same `--fold`/`--seed` to every later command.

## 3. Evaluate on MS3SEG (expert labels)

```bash
python segmamba_v2/predict.py  --fold 1 --split val
python segmamba_v2/evaluate.py --fold 1 --split val --tune     # minimum MS lesion size
python segmamba_v2/predict.py  --fold 1 --split test           # official 20 test patients
python segmamba_v2/evaluate.py --fold 1 --split test
```
- Predictions go to `prediction_results/<run>/{val,test}/`, on the native FLAIR grid with its header:
  - `<id>.nii.gz`: label map (0 background, 1 ventricles, 2 normal WMH, 3 MS lesion);
  - `<id>_prob_<class>.nii.gz`: per-class probability.
- `evaluate.py` scores every class against its expert mask:
  - Dice, volumes and their absolute difference, HD95;
  - for MS lesions also lesion-wise sensitivity, precision and F1.

  Per class it prints the summary and the volume agreement (Pearson r, ICC(2,1), Bland–Altman), then the mean Dice over the three classes, as in the paper's multi-class table. It writes `metrics.csv` with one row per patient and class.
- For context, the MS3SEG paper reports 5-fold Dice for U-Net (4-class) of 0.89 (ventricles), 0.59 (normal WMH) and 0.67 (MS lesion). Its models are 2D, so the protocols differ and the numbers are not strictly comparable.

## 4. Test on the SVD data (reference = LST-AI)

Input is the SVD layout from `Shohag_vai_lab_data/sample_dataset/README.md`: `data/LSTAI_inputs/SVD_###/{flair,t1}.nii.gz` and `output/LSTAI_results/SVD_###/space-flair_seg-lst.nii.gz`.

Step 1 needs FSL and can run wherever FSL and the data are (e.g. under WSL). Steps 2 and 3 run on the GPU PC.

```bash
# 1. FSL: prepare like MS3SEG (T1 registered to FLAIR with flirt, FLAIR brain mask with bet -f 0.5)
python segmamba_v2/svd_flair.py --inputs /path/to/data/LSTAI_inputs --out data/svd_flair
#    -> look at the cases flagged in data/svd_flair/qc.csv (unusual brain-mask volume) and at a few T1-on-FLAIR overlays

# 2. GPU: predict; label maps land on the native FLAIR grid, the same grid as LST-AI's masks
python segmamba_v2/predict.py --fold 1 --input_dir data/svd_flair

# 3. compare with LST-AI (normal WMH + MS lesion merged; add --pred_labels 3 for MS lesions only)
python segmamba_v2/evaluate.py --pred_dir prediction_results/segmamba_v2_ms3seg_fold1/svd_flair \
    --gt_dir /path/to/output/LSTAI_results
```
- **Choose the comparison deliberately.** LST-AI marks one binary "lesion" class. In MS3SEG terms, SVD white-matter hyperintensities may fall under normal WMH (periventricular caps and bands), under MS-like lesions, or under both. So report both comparisons:
  - the default merge, normal WMH + MS lesion;
  - `--pred_labels 3`, MS lesion only.
- These numbers measure **agreement with LST-AI**, not accuracy. Read them next to the MS3SEG test results from step 3.

## Notes

- **Spacing.** The network sees 1 mm slices at native in-plane spacing (0.8984 mm), set by `config.out_spacing`, which is in (z, y, x) order. Resampling in-plane to 1 mm as well and back cost normal WMH about 7% Dice even for a perfect prediction. With the current setting, the round trip from native grid to network spacing and back is lossless (Dice 1.000 per class).
- **T1 coverage.** In a few MS3SEG patients, T1 covers 18 slices against FLAIR's 20, so the lowest or highest FLAIR slice has no T1 signal. The middle slices are well aligned, so no patient is excluded.
- **Inference needs the GPU.** The Mamba layers use compiled CUDA kernels even for inference.
- **External data must look like MS3SEG.** `predict.py --input_dir` expects each case folder to hold `flair.nii.gz`, `t1.nii.gz` on the FLAIR grid and `brain_mask.nii.gz`. `svd_flair.py` produces exactly that.
