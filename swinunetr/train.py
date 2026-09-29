"""Train Swin UNETR on BraTS2020 with the same data, split and pipeline as 3_train.py.

Run from anywhere:  python swinunetr/train.py
"""
import config as cfg  # must be first: sets sys.path, cwd and CUDA allocator config

import json
import os

import numpy as np
import torch
import torch.nn as nn
from monai.losses import DiceCELoss
from monai.utils import set_determinism

from light_training.dataloading.dataset import get_train_val_test_loader_from_train
from light_training.evaluation.metric import dice
from light_training.trainer import Trainer
from light_training.utils.files_helper import save_new_model_and_delete_last

set_determinism(123)


class SwinUNETRTrainer(Trainer):
    def __init__(self, env_type, max_epochs, batch_size, device="cpu", val_every=1, num_gpus=1, logdir="./logs/", master_ip='localhost', master_port=17750, training_script="train.py"):
        super().__init__(env_type, max_epochs, batch_size, device, val_every, num_gpus, logdir, master_ip, master_port, training_script)
        self.augmentation = cfg.augmentation
        self.patch_size = cfg.roi_size
        self.train_process = 18
        self.best_mean_dice = 0.0

        self.model = cfg.build_model()

        if cfg.optimizer_name == "adamw":
            self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
            self.scheduler_type = "cosine_with_warmup"
            self.warmup = cfg.warmup_ratio
        elif cfg.optimizer_name == "sgd":
            # identical to SegMamba
            self.optimizer = torch.optim.SGD(self.model.parameters(), lr=1e-2, weight_decay=3e-5,
                                             momentum=0.99, nesterov=True)
            self.scheduler_type = "poly"
        else:
            raise ValueError(f"unknown optimizer {cfg.optimizer_name}")

        if cfg.loss_name == "ce":
            self.loss_fn = nn.CrossEntropyLoss()
        elif cfg.loss_name == "dice_ce":
            dice_ce = DiceCELoss(to_onehot_y=True, softmax=True)
            self.loss_fn = lambda pred, label: dice_ce(pred, label[:, None])
        else:
            raise ValueError(f"unknown loss {cfg.loss_name}")

    def get_input(self, batch):
        image = batch["data"]
        label = batch["seg"][:, 0].long()
        return image, cfg.map_brats_labels(label)

    def training_step(self, batch):
        image, label = self.get_input(batch)
        pred = self.model(image)
        loss = self.loss_fn(pred, label)
        self.log("training_loss", loss, step=self.global_step)
        return loss

    def cal_metric(self, gt, pred):
        # same as 3_train.py
        if pred.sum() > 0 and gt.sum() > 0:
            return dice(pred, gt)
        elif gt.sum() == 0 and pred.sum() == 0:
            return 1.0
        return 0.0

    def validation_step(self, batch):
        image, label = self.get_input(batch)
        output = self.model(image).argmax(dim=1)

        output = cfg.convert_labels(output[:, None]).cpu().numpy()
        target = cfg.convert_labels(label[:, None]).cpu().numpy()

        return [self.cal_metric(target[:, i], output[:, i]) for i in range(3)]

    def validation_end(self, val_outputs):
        tc, wt, et = val_outputs[0].mean(), val_outputs[1].mean(), val_outputs[2].mean()
        print(f"dices is {tc, wt, et}")
        mean_dice = (tc + wt + et) / 3

        self.log("tc", tc, step=self.epoch)
        self.log("wt", wt, step=self.epoch)
        self.log("et", et, step=self.epoch)
        self.log("mean_dice", mean_dice, step=self.epoch)

        if mean_dice > self.best_mean_dice:
            self.best_mean_dice = mean_dice
            save_new_model_and_delete_last(self.model,
                                           os.path.join(cfg.model_save_path, f"best_model_{mean_dice:.4f}.pt"),
                                           delete_symbol="best_model")

        save_new_model_and_delete_last(self.model,
                                       os.path.join(cfg.model_save_path, f"final_model_{mean_dice:.4f}.pt"),
                                       delete_symbol="final_model")

        if (self.epoch + 1) % 100 == 0:
            torch.save(self.model.state_dict(),
                       os.path.join(cfg.model_save_path, f"tmp_model_ep{self.epoch}_{mean_dice:.4f}.pt"))

        if torch.cuda.is_available():
            print(f"peak GPU memory so far: {torch.cuda.max_memory_allocated() / 1024**3:.2f} GB")
        print(f"mean_dice is {mean_dice}")


def save_split(train_ds, val_ds, test_ds):
    """Record the split so predict.py can verify it matches (the split relies on glob order)."""
    split = {"train": cfg.split_names(train_ds), "validation": cfg.split_names(val_ds), "test": cfg.split_names(test_ds)}
    os.makedirs(cfg.logdir, exist_ok=True)
    if os.path.exists(cfg.split_file):
        with open(cfg.split_file) as f:
            if json.load(f) != split:
                print(f"WARNING: data split differs from the one saved in {cfg.split_file}; overwriting it.")
    with open(cfg.split_file, "w") as f:
        json.dump(split, f, indent=1)


def ensure_non_empty_split(train_ds, val_ds, test_ds):
    counts = {
        "train": len(train_ds),
        "validation": len(val_ds),
        "test": len(test_ds),
    }
    if min(counts.values()) > 0:
        return

    raise RuntimeError(
        "Empty dataset split detected: "
        f"train={counts['train']}, val={counts['validation']}, test={counts['test']}. "
        f"Configured data_dir={cfg.data_dir}. "
        "Expected BraTS preprocessed files (*.npz, *.pkl, *.npy, *_seg.npy) under that folder."
    )


if __name__ == "__main__":
    trainer = SwinUNETRTrainer(env_type=cfg.env,
                               max_epochs=cfg.max_epoch,
                               batch_size=cfg.batch_size,
                               device=cfg.device,
                               logdir=cfg.logdir,
                               val_every=cfg.val_every,
                               num_gpus=cfg.num_gpus,
                               master_port=17760,
                               training_script=__file__)

    train_ds, val_ds, test_ds = get_train_val_test_loader_from_train(cfg.data_dir)
    ensure_non_empty_split(train_ds, val_ds, test_ds)
    save_split(train_ds, val_ds, test_ds)

    trainer.train(train_dataset=train_ds, val_dataset=val_ds)
