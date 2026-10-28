"""Train SegMamba v2 on MSLesSeg (FLAIR + T1 -> lesion) for one patient-level fold.

Run from anywhere (after preprocess.py):
    python segmamba_v2/train.py --fold 0              (-> logs/segmamba_v2_ms_fold0/)
    python segmamba_v2/train.py --fold 0 --seed 7     (-> logs/segmamba_v2_ms_fold0_seed7/)
Validation every 2 epochs on 100 random lesion-containing patches of the fold's validation scans;
the best patch Dice is saved as logs/<run>/model/best_model_<dice>.pt.
"""
import config as cfg  # must be first: sets sys.path, cwd and CUDA allocator config

import csv
import json
import os

import numpy as np
import torch
from monai.utils import set_determinism

import light_training.trainer as lt_trainer
from light_training.dataloading.dataset import MedicalDataset
from light_training.evaluation.metric import dice
from light_training.trainer import Trainer
from light_training.utils.files_helper import save_new_model_and_delete_last
from losses import DeepSupervisionLoss, DiceBCELoss

# Seeding as in the BraTS v2 run: 123 here (model init), then Trainer.train() re-seeds with 42
# (data/augmentation). Another --seed N shifts both by N - 123 so repeat runs actually differ.
set_determinism(cfg.seed)
_seed_offset = cfg.seed - cfg.default_seed
lt_trainer.set_determinism = lambda seed=None, **kwargs: set_determinism(None if seed is None else seed + _seed_offset, **kwargs)


class SegMambaMSTrainer(Trainer):
    def __init__(self, env_type, max_epochs, batch_size, device="cpu", val_every=1, num_gpus=1, logdir="./logs/", master_ip='localhost', master_port=17750, training_script="train.py"):
        super().__init__(env_type, max_epochs, batch_size, device, val_every, num_gpus, logdir, master_ip, master_port, training_script)
        self.augmentation = cfg.augmentation
        self.patch_size = cfg.roi_size
        self.train_process = cfg.train_process
        self.best_mean_dice = 0.0
        self.epoch_losses = []

        self.model = cfg.build_model()

        if cfg.optimizer_name == "sgd":
            self.optimizer = torch.optim.SGD(self.model.parameters(), lr=cfg.sgd_lr, weight_decay=3e-5,
                                             momentum=0.99, nesterov=True)
            self.scheduler_type = "poly"
        elif cfg.optimizer_name == "adamw":
            self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=cfg.adamw_lr, weight_decay=cfg.adamw_weight_decay)
            self.scheduler_type = "cosine_with_warmup"
            self.warmup = cfg.warmup_ratio
        else:
            raise ValueError(f"unknown optimizer {cfg.optimizer_name}")

        self.loss_fn = DeepSupervisionLoss(DiceBCELoss(batch_dice=cfg.batch_dice))

    def get_input(self, batch):
        # seg: 1 = lesion, 0 = background, -1 = outside the brain crop (padding) -> background
        return batch["data"], (batch["seg"] > 0).float()  # (B, 2, D, H, W), (B, 1, D, H, W)

    def training_step(self, batch):
        image, target = self.get_input(batch)
        loss = self.loss_fn(self.model(image), target)
        self.epoch_losses.append(loss.item())
        self.log("training_loss", loss, step=self.global_step)
        return loss

    def train_epoch(self, epoch):
        self.epoch_losses = []
        super().train_epoch(epoch)
        # Mean loss over the epoch (not just the last batch), so the curve is not lost with the terminal.
        lr = self.optimizer.param_groups[0]["lr"]
        self.log("epoch_train_loss", float(np.mean(self.epoch_losses)), step=epoch)
        append_csv(os.path.join(cfg.logdir, "train_log.csv"), ["epoch", "train_loss", "lr"],
                   [epoch, float(np.mean(self.epoch_losses)), lr])

    def cal_metric(self, gt, pred):
        if pred.sum() > 0 and gt.sum() > 0:
            return dice(pred, gt)
        elif gt.sum() == 0 and pred.sum() == 0:
            return 1.0
        return 0.0

    def validation_step(self, batch):
        image, target = self.get_input(batch)
        output = (torch.sigmoid(self.model(image).float()) > cfg.threshold).cpu().numpy()
        return self.cal_metric(target.cpu().numpy(), output)

    def validation_end(self, val_outputs):
        mean_dice = float(val_outputs.mean())
        print(f"lesion dice (validation patches) is {mean_dice:.4f}")

        self.log("lesion_dice", mean_dice, step=self.epoch)
        append_csv(os.path.join(cfg.logdir, "val_log.csv"), ["epoch", "lesion_dice"], [self.epoch, mean_dice])

        if mean_dice > self.best_mean_dice:
            self.best_mean_dice = mean_dice
            save_new_model_and_delete_last(self.model,
                                           os.path.join(cfg.model_save_path, f"best_model_{mean_dice:.4f}.pt"),
                                           delete_symbol="best_model")

        save_new_model_and_delete_last(self.model,
                                       os.path.join(cfg.model_save_path, f"final_model_{mean_dice:.4f}.pt"),
                                       delete_symbol="final_model")

        if (self.epoch + 1) % 50 == 0:
            torch.save(self.model.state_dict(),
                       os.path.join(cfg.model_save_path, f"tmp_model_ep{self.epoch}_{mean_dice:.4f}.pt"))

        if torch.cuda.is_available():
            print(f"peak GPU memory so far: {torch.cuda.max_memory_allocated() / 1024**3:.2f} GB")


def append_csv(path, header, row):
    new = not os.path.exists(path)
    with open(path, "a", newline="") as f:
        writer = csv.writer(f)
        if new:
            writer.writerow(header)
        writer.writerow(row)


def load_datasets():
    """MedicalDatasets of the fold's train and val scans (preprocessed .npz in cfg.data_dir)."""
    train_cases, val_cases = cfg.split_cases("train"), cfg.split_cases("val")
    missing = [c for c in train_cases + val_cases if not os.path.exists(os.path.join(cfg.data_dir, f"{c}.npz"))]
    if missing:
        raise FileNotFoundError(f"{len(missing)} preprocessed cases missing in {cfg.data_dir} (e.g. {missing[:3]}); "
                                f"run segmamba_v2/preprocess.py")
    print(f"fold {cfg.fold}: {len(train_cases)} train scans, {len(val_cases)} val scans")

    os.makedirs(cfg.logdir, exist_ok=True)
    with open(os.path.join(cfg.logdir, "data_split.json"), "w") as f:
        json.dump({"fold": cfg.fold, "train": train_cases, "validation": val_cases}, f, indent=1)

    to_paths = lambda cases: [os.path.join(cfg.data_dir, f"{c}.npz") for c in cases]
    return MedicalDataset(to_paths(train_cases)), MedicalDataset(to_paths(val_cases))


if __name__ == "__main__":
    for log_file in ("train_log.csv", "val_log.csv"):
        if os.path.exists(os.path.join(cfg.logdir, log_file)):
            raise FileExistsError(f"{cfg.logdir}/{log_file} exists; move the old run away or use another --seed")

    train_ds, val_ds = load_datasets()

    trainer = SegMambaMSTrainer(env_type=cfg.env,
                                max_epochs=cfg.max_epoch,
                                batch_size=cfg.batch_size,
                                device=cfg.device,
                                logdir=cfg.logdir,
                                val_every=cfg.val_every,
                                num_gpus=cfg.num_gpus,
                                master_port=17762,
                                training_script=__file__)

    trainer.train(train_dataset=train_ds, val_dataset=val_ds)
