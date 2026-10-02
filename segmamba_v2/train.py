"""Train SegMamba v2 (Phase A recipe) on BraTS2020 with the same data and split as 3_train.py.

Run from anywhere:  python segmamba_v2/train.py              (seed 123 -> logs/segmamba_v2/)
                    python segmamba_v2/train.py --seed 7     (-> logs/segmamba_v2_seed7/)
"""
import config as cfg  # must be first: sets sys.path, cwd and CUDA allocator config

import csv
import json
import os

import numpy as np
import torch
from monai.utils import set_determinism

import light_training.trainer as lt_trainer
from light_training.dataloading.dataset import get_train_val_test_loader_from_train
from light_training.evaluation.metric import dice
from light_training.trainer import Trainer
from light_training.utils.files_helper import save_new_model_and_delete_last
from losses import DeepSupervisionLoss, DiceBCELoss

# Same seeding as 3_train.py and swinunetr/train.py: 123 here (model init), then Trainer.train()
# re-seeds with 42 (data/augmentation). The default --seed 123 reproduces exactly that; another
# --seed N shifts both by N - 123 so repeat runs actually differ.
set_determinism(cfg.seed)
_seed_offset = cfg.seed - cfg.default_seed
lt_trainer.set_determinism = lambda seed=None, **kwargs: set_determinism(None if seed is None else seed + _seed_offset, **kwargs)


class SegMambaV2Trainer(Trainer):
    def __init__(self, env_type, max_epochs, batch_size, device="cpu", val_every=1, num_gpus=1, logdir="./logs/", master_ip='localhost', master_port=17750, training_script="train.py"):
        super().__init__(env_type, max_epochs, batch_size, device, val_every, num_gpus, logdir, master_ip, master_port, training_script)
        self.augmentation = cfg.augmentation
        self.patch_size = cfg.roi_size
        self.train_process = 18
        self.best_mean_dice = 0.0
        self.epoch_losses = []

        self.model = cfg.build_model()

        if cfg.optimizer_name == "sgd":
            # same optimizer as 3_train.py; poly now decays to 0 over max_epoch = 250
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
        image = batch["data"]
        label = cfg.map_brats_labels(batch["seg"][:, 0].long())
        return image, cfg.convert_labels(label[:, None])  # (B, 3, D, H, W) TC/WT/ET

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
        # same as 3_train.py
        if pred.sum() > 0 and gt.sum() > 0:
            return dice(pred, gt)
        elif gt.sum() == 0 and pred.sum() == 0:
            return 1.0
        return 0.0

    def validation_step(self, batch):
        image, target = self.get_input(batch)
        regions = torch.sigmoid(self.model(image).float()) > cfg.threshold
        output = cfg.convert_labels(cfg.regions_to_labels(regions)).cpu().numpy()
        target = target.cpu().numpy()
        return [self.cal_metric(target[:, i], output[:, i]) for i in range(3)]

    def validation_end(self, val_outputs):
        tc, wt, et = val_outputs[0].mean(), val_outputs[1].mean(), val_outputs[2].mean()
        print(f"dices is {tc, wt, et}")
        mean_dice = (tc + wt + et) / 3

        self.log("tc", tc, step=self.epoch)
        self.log("wt", wt, step=self.epoch)
        self.log("et", et, step=self.epoch)
        self.log("mean_dice", mean_dice, step=self.epoch)
        append_csv(os.path.join(cfg.logdir, "val_log.csv"), ["epoch", "tc", "wt", "et", "mean_dice"],
                   [self.epoch, float(tc), float(wt), float(et), float(mean_dice)])

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
        print(f"mean_dice is {mean_dice}")


def append_csv(path, header, row):
    new = not os.path.exists(path)
    with open(path, "a", newline="") as f:
        writer = csv.writer(f)
        if new:
            writer.writerow(header)
        writer.writerow(row)


def save_split(train_ds, val_ds, test_ds):
    """Record the split so predict.py can verify it matches (the split relies on glob order)."""
    if min(len(train_ds), len(val_ds), len(test_ds)) == 0:
        raise RuntimeError(f"Empty dataset split (train={len(train_ds)}, val={len(val_ds)}, test={len(test_ds)}); "
                           f"check data_dir={cfg.data_dir}")
    split = {"train": cfg.split_names(train_ds), "validation": cfg.split_names(val_ds), "test": cfg.split_names(test_ds)}

    checked = False
    for ref_file in cfg.reference_split_files:
        if not os.path.exists(ref_file):
            continue
        with open(ref_file) as f:
            if json.load(f) != split:
                raise RuntimeError(f"data split differs from {ref_file}; results would not be comparable. "
                                   f"Check that {cfg.data_dir} holds the same files as for that run.")
        print(f"data split matches {ref_file}")
        checked = True
    if not checked:
        print(f"WARNING: none of {cfg.reference_split_files} found; cannot verify the split against earlier runs.")

    os.makedirs(cfg.logdir, exist_ok=True)
    with open(cfg.split_file, "w") as f:
        json.dump(split, f, indent=1)


if __name__ == "__main__":
    for log_file in ("train_log.csv", "val_log.csv"):
        if os.path.exists(os.path.join(cfg.logdir, log_file)):
            raise FileExistsError(f"{cfg.logdir}/{log_file} exists; move the old run away or use another --seed")

    trainer = SegMambaV2Trainer(env_type=cfg.env,
                                max_epochs=cfg.max_epoch,
                                batch_size=cfg.batch_size,
                                device=cfg.device,
                                logdir=cfg.logdir,
                                val_every=cfg.val_every,
                                num_gpus=cfg.num_gpus,
                                master_port=17762,
                                training_script=__file__)

    train_ds, val_ds, test_ds = get_train_val_test_loader_from_train(cfg.data_dir)
    save_split(train_ds, val_ds, test_ds)

    trainer.train(train_dataset=train_ds, val_dataset=val_ds)
