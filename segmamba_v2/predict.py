"""Predict a split with SegMamba v2 (sliding window + 8x mirror TTA, same as 4_predict.py).

Run:  python segmamba_v2/predict.py --split val     -> prediction_results/segmamba_v2_val/  (for tuning post-processing)
      python segmamba_v2/predict.py --split test    -> prediction_results/segmamba_v2/
      add --ckpt path/to/model.pt to choose a checkpoint (default: best_model_*.pt), --seed N for another run
Output: <case>.nii.gz with TC/WT/ET channels, same format as 4_predict.py, no post-processing.
"""
import config as cfg  # must be first

import argparse
import glob
import json
import os

import torch
from monai.inferers import SlidingWindowInferer
from monai.utils import set_determinism

from light_training.dataloading.dataset import get_train_val_test_loader_from_train
from light_training.evaluation.metric import dice
from light_training.prediction import Predictor
from light_training.trainer import Trainer

set_determinism(123)


def find_checkpoint(ckpt):
    if ckpt is not None:
        return ckpt
    candidates = glob.glob(os.path.join(cfg.model_save_path, "best_model_*.pt"))
    if not candidates:
        raise FileNotFoundError(f"no best_model_*.pt in {cfg.model_save_path}; pass --ckpt")
    return max(candidates, key=lambda p: float(os.path.basename(p)[len("best_model_"):-3]))


def load_state_dict(path):
    sd = torch.load(path, map_location="cpu")
    if "module" in sd:
        sd = sd["module"]
    return {(k[7:] if k.startswith("module") else k): v for k, v in sd.items()}


def check_split(ds, split):
    if len(ds) == 0:
        raise RuntimeError(f"Empty {split} split; check data_dir={cfg.data_dir}")
    if not os.path.exists(cfg.split_file):
        print(f"WARNING: {cfg.split_file} not found, cannot verify the {split} split.")
        return
    with open(cfg.split_file) as f:
        saved = json.load(f)["validation" if split == "val" else "test"]
    if saved != cfg.split_names(ds):
        raise RuntimeError(f"{split} split differs from the one used during training ({cfg.split_file}).")


class SegMambaV2Predictor(Trainer):
    def __init__(self, ckpt, save_dir, env_type, max_epochs, batch_size, device="cpu", val_every=1, num_gpus=1, logdir="./logs/", master_ip='localhost', master_port=17750, training_script="train.py"):
        super().__init__(env_type, max_epochs, batch_size, device, val_every, num_gpus, logdir, master_ip, master_port, training_script)
        self.patch_size = cfg.roi_size
        self.augmentation = False
        self.save_dir = save_dir

        print(f"loading {ckpt}")
        self.net = cfg.build_model()
        self.net.load_state_dict(load_state_dict(ckpt))
        self.net.eval()  # eval mode -> only the full-resolution output, no deep-supervision heads

        window_infer = SlidingWindowInferer(roi_size=cfg.roi_size,
                                            sw_batch_size=cfg.sw_batch_size,
                                            overlap=cfg.sw_overlap,
                                            progress=True,
                                            mode="gaussian")
        self.predictor = Predictor(window_infer=window_infer, mirror_axes=cfg.mirror_axes)
        os.makedirs(save_dir, exist_ok=True)

    def validation_step(self, batch):
        image = batch["data"]
        properties = batch["properties"]
        label = cfg.convert_labels(cfg.map_brats_labels(batch["seg"]))[0]

        logits = self.predictor.maybe_mirror_and_predict(image, self.net, device=cfg.device)
        logits = self.predictor.predict_raw_probability(logits, properties=properties)
        regions = torch.sigmoid(logits.float()) > cfg.threshold
        model_output = cfg.convert_labels(cfg.regions_to_labels(regions, dim=0), dim=0)

        dices = [dice(model_output[i].cpu().numpy(), label[i].cpu().numpy()) for i in range(3)]
        print(f"{properties['name'][0]} TC/WT/ET dice (cropped space): {dices}")

        model_output = self.predictor.predict_noncrop_probability(model_output, properties)
        self.predictor.save_to_nii(model_output,
                                   raw_spacing=[1, 1, 1],
                                   case_name=properties['name'][0],
                                   save_dir=self.save_dir)
        torch.cuda.empty_cache()
        return dices


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=["val", "test"], default="test")
    parser.add_argument("--ckpt", type=str, default=None)
    parser.add_argument("--seed", type=int, default=123)  # parsed in config.py; selects the run folder
    args = parser.parse_args()

    train_ds, val_ds, test_ds = get_train_val_test_loader_from_train(cfg.data_dir)
    ds = val_ds if args.split == "val" else test_ds
    check_split(ds, args.split)

    save_dir = cfg.pred_dir(args.split)
    predictor = SegMambaV2Predictor(ckpt=find_checkpoint(args.ckpt),
                                    save_dir=save_dir,
                                    env_type=cfg.env,
                                    max_epochs=cfg.max_epoch,
                                    batch_size=cfg.batch_size,
                                    device=cfg.device,
                                    logdir="",
                                    val_every=cfg.val_every,
                                    num_gpus=cfg.num_gpus,
                                    master_port=17763,
                                    training_script=__file__)

    mean_dices, _ = predictor.validation_single_gpu(ds)
    print(f"mean TC/WT/ET dice (cropped space): {[float(d) for d in mean_dices]}")
    print(f"saved to {save_dir}")
