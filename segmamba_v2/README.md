# SegMamba v2 on MSLesSeg (FLAIR + T1 → MS lesions)

Same network and training recipe as SegMamba v2 on BraTS (branch `main`). The changes are for MS lesions and for testing on data that has only FLAIR and T1, such as the lab's SVD cohort:

| | SegMamba v2 on BraTS (`main`) | This branch |
|---|---|---|
| Input | FLAIR, T1, T1ce, T2 (4 channels) | **FLAIR + T1** (2 channels), so the model runs on SVD scans without T2 |
| Output | TC/WT/ET (3 sigmoid channels) | **lesion** (1 sigmoid channel) |
| Split | random 70/10/20 per file | **patient-level 5-fold CV** over the 53 MSLesSeg train patients; the official test set (22 patients) is held out |
| Normalization | z-score over the cropped box | z-score **inside each channel's brain mask**, 0 outside |
| Excluded scans | none | 7 train scans with a broken or misregistered FLAIR/T1 (`config.excluded`) |
| Post-processing | ET small-component rules | remove predicted lesions below a size tuned on validation |
| Saved predictions | spacing only (orientation lost) | the input FLAIR's full header, plus a probability map |
| Unchanged | | SGD 1e-2 + poly over 250 epochs, BCE + soft Dice, deep supervision, 128³ patches, batch 2, augmentation, sliding window + 8× mirror TTA |

All settings are in `config.py`.

## 0. Set up the GPU PC (once)

1. Check out branch `msles2025` and install the environment: see the top-level README (CUDA extensions + `requirements.txt`).
2. Copy the `MSLesSeg Dataset` folder there. It is the one holding `train/`, `test/` and `info_dataset/`, and comes from `MSLesSeg_2025/MSLesSeg Dataset.zip` (1.4 GB).
3. Check that the CUDA extensions work and that training fits in memory (about a minute):
   ```bash
   python segmamba_v2/check_gpu.py
   ```
   It prints the peak memory of 3 training steps (batch 2, 128³) and of one full-volume inference. If training is close to 24 GB, set `batch_size = 1` in `config.py`.

## 1. Preprocess MSLesSeg (about 1 minute)

```bash
python segmamba_v2/preprocess.py --mslesseg_dir "/path/to/MSLesSeg Dataset"
```
This writes `data/mslesseg_fullres/` (set `MSLESSEG_DATA_DIR` to put it elsewhere):
- `cases.json`: every scan with its files, lesion volume and, where applicable, the reason it is excluded.
- `splits.json`: the test cases and the 5 folds. Folds are patient-level: all timepoints of a patient are in the same fold. They are stratified by lesion load.
- `<case>.npz/.pkl`: the 86 fold scans, cropped and normalized.

Plan for about 4 GB of disk: training unpacks the `.npz` files to `.npy` next to them.

The fold sizes printed by the script are, for validation: 15, 20, 12, 25 and 14 scans (10–11 patients each).

## 2. Train (about 10 h per fold on an RTX 4090)

```bash
python segmamba_v2/train.py --fold 0
tensorboard --logdir logs          # optional, to watch the curves
```
- Output goes to `logs/segmamba_v2_ms_fold0/`: `train_log.csv`, `val_log.csv`, `data_split.json` and `model/best_model_<dice>.pt`.
- Every 2 epochs the model is scored on 100 random lesion-containing patches of the fold's validation scans, and the best checkpoint is kept. This patch Dice selects checkpoints; it is not the case-level Dice reported in step 3.
- Start with fold 0. Folds 1–4 (`--fold 1` …) give a cross-validated estimate if there is GPU time for them.
- `--seed N` repeats a fold with another seed, into `logs/segmamba_v2_ms_fold0_seedN/`. Pass the same `--fold`/`--seed` to every later command.

## 3. Evaluate on MSLesSeg (expert labels)

```bash
python segmamba_v2/predict.py  --fold 0 --split val            # validation scans of fold 0
python segmamba_v2/evaluate.py --fold 0 --split val --tune     # choose the minimum lesion size
python segmamba_v2/predict.py  --fold 0 --split test           # official test set, small lesions removed
python segmamba_v2/evaluate.py --fold 0 --split test
```
- `--tune` picks the minimum lesion size (0–50 voxels) that maximizes lesion-wise F1 on validation. It saves the choice to `logs/<run>/postprocessing.json`, and from then on `predict.py` applies it.
- Predictions go to `prediction_results/<run>/{val,test}/`:
  - `<case>.nii.gz` is the binary mask;
  - `<case>_prob.nii.gz` is the probability map;
  - both are on the input FLAIR grid with its header.
- `evaluate.py` prints a summary table and writes `metrics.csv` next to the predictions. Each case gets:
  - Dice;
  - lesion-wise sensitivity, precision and F1 (lesions are 26-connected components; a lesion counts as found if it shares at least one voxel);
  - reference and predicted volume, and their absolute difference;
  - HD95.

  Across cases it adds volume agreement: Pearson r, ICC(2,1) and Bland–Altman.

## 4. Test on the SVD data (reference = LST-AI)

Input is the SVD layout from `Shohag_vai_lab_data/sample_dataset/README.md`. The scripts use `data/LSTAI_inputs/SVD_###/{flair,t1}.nii.gz`, the same images LST-AI used, so subjects without a T1 are already left out. The reference is `output/LSTAI_results/SVD_###/space-flair_seg-lst.nii.gz`.

Steps 1 and 3 need FSL. They can run on any machine that has FSL and the data (for example under WSL), and steps 2 and 4 on the GPU PC.

```bash
# 1. FSL: same registration + skull stripping as MSLesSeg (flirt to MNI152 1 mm, then bet)
python segmamba_v2/svd_mni.py to_mni --inputs /path/to/data/LSTAI_inputs --out data/svd_mni
#    -> look at the cases flagged in data/svd_mni/qc.csv (FLAIR and T1 brain volumes disagree) before going on

# 2. GPU: predict in MNI space
python segmamba_v2/predict.py --fold 0 --input_dir data/svd_mni

# 3. FSL: map the probability maps back to native FLAIR space and threshold at 0.5
python segmamba_v2/svd_mni.py to_native --inputs /path/to/data/LSTAI_inputs --mni_dir data/svd_mni \
    --pred_dir prediction_results/segmamba_v2_ms_fold0/svd_mni \
    --out prediction_results/segmamba_v2_ms_fold0/svd_native

# 4. compare with LST-AI, subject by subject, in native FLAIR space
python segmamba_v2/evaluate.py --pred_dir prediction_results/segmamba_v2_ms_fold0/svd_native \
    --gt_dir /path/to/output/LSTAI_results
```
- Step 3 writes `svd_native/SVD_###/space-flair_seg-segmamba.nii.gz` and `space-flair_prob-segmamba.nii.gz`. They sit on the same grid as LST-AI's `space-flair_seg-lst.nii.gz`, so they can be opened together.
- LST-AI is itself an automatic model, so these numbers measure **agreement with LST-AI**, not accuracy. Read them next to the MSLesSeg test results from step 3.

## Notes

- **Excluded scans.** QC of the preprocessed MSLesSeg data found these problems:
  - P49_T2: the T1 is empty;
  - P44_T1, P45_T1, P53_T1: the FLAIR was registered about 1.6× too large;
  - P12_T1, P12_T3, P12_T4: the T1 is misaligned with FLAIR.

  The lesion masks follow FLAIR in all of them. These scans are in no fold. The official test set has no such problems and is used in full.
- **Inference needs the GPU.** The Mamba layers use the compiled CUDA kernels even for inference, so `predict.py` cannot run on a CPU (unlike LST-AI).
- **External data must look like MSLesSeg.** `predict.py --input_dir` expects images in MNI152 1 mm, skull-stripped, with FLAIR and T1 on the same grid. `svd_mni.py to_mni` produces exactly that. Other spacings are resampled to 1 mm, but other spaces (native, not skull-stripped) will not work.
