"""Predict the test split with Swin UNETR, exactly like 4_predict.py does for SegMamba.

Run:  python swinunetr/predict.py                      (uses logs/swinunetr/model/best_model_*.pt)
      python swinunetr/predict.py --ckpt path/to/model.pt
Output: prediction_results/swinunetr/<case>.nii.gz  (TC/WT/ET channels, same format as SegMamba)
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


def check_split(test_ds):
    if len(test_ds) == 0:
        raise RuntimeError(
            "Empty test split detected. "
            f"Configured data_dir={cfg.data_dir}. "
            "Expected BraTS preprocessed files (*.npz, *.pkl, *.npy, *_seg.npy) under that folder."
        )

    if not os.path.exists(cfg.split_file):
        print(f"WARNING: {cfg.split_file} not found, cannot verify the test split.")
        return
    with open(cfg.split_file) as f:
        saved = json.load(f)["test"]
    if saved != cfg.split_names(test_ds):
        raise RuntimeError("test split differs from the one used during training "
                           f"({cfg.split_file}); the comparison would not be valid.")


class SwinUNETRPredictor(Trainer):
    def __init__(self, ckpt, env_type, max_epochs, batch_size, device="cpu", val_every=1, num_gpus=1, logdir="./logs/", master_ip='localhost', master_port=17750, training_script="train.py"):
        super().__init__(env_type, max_epochs, batch_size, device, val_every, num_gpus, logdir, master_ip, master_port, training_script)
        self.patch_size = cfg.roi_size
        self.augmentation = False

        print(f"loading {ckpt}")
        # Gradient checkpointing is irrelevant without grads; build the model once, not per case.
        self.net = cfg.build_model()
        self.net.load_state_dict(load_state_dict(ckpt))
        self.net.eval()

        window_infer = SlidingWindowInferer(roi_size=cfg.roi_size,
                                            sw_batch_size=cfg.sw_batch_size,
                                            overlap=cfg.sw_overlap,
                                            progress=True,
                                            mode="gaussian")
        self.predictor = Predictor(window_infer=window_infer, mirror_axes=cfg.mirror_axes)
        os.makedirs(cfg.save_path, exist_ok=True)

    def validation_step(self, batch):
        image = batch["data"]
        properties = batch["properties"]
        label = cfg.convert_labels(cfg.map_brats_labels(batch["seg"]))[0]

        model_output = self.predictor.maybe_mirror_and_predict(image, self.net, device=cfg.device)
        model_output = self.predictor.predict_raw_probability(model_output, properties=properties)
        model_output = cfg.convert_labels(model_output.argmax(dim=0)[None], dim=0)

        dices = [dice(model_output[i].cpu().numpy(), label[i].cpu().numpy()) for i in range(3)]
        print(f"{properties['name'][0]} TC/WT/ET dice (cropped space): {dices}")

        model_output = self.predictor.predict_noncrop_probability(model_output, properties)
        self.predictor.save_to_nii(model_output,
                                   raw_spacing=[1, 1, 1],
                                   case_name=properties['name'][0],
                                   save_dir=cfg.save_path)
        torch.cuda.empty_cache()
        return dices


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, default=None)
    args = parser.parse_args()

    train_ds, val_ds, test_ds = get_train_val_test_loader_from_train(cfg.data_dir)
    check_split(test_ds)

    predictor = SwinUNETRPredictor(ckpt=find_checkpoint(args.ckpt),
                                   env_type=cfg.env,
                                   max_epochs=cfg.max_epoch,
                                   batch_size=cfg.batch_size,
                                   device=cfg.device,
                                   logdir="",
                                   val_every=cfg.val_every,
                                   num_gpus=cfg.num_gpus,
                                   master_port=17761,
                                   training_script=__file__)

    mean_dices, _ = predictor.validation_single_gpu(test_ds)
    print(f"mean TC/WT/ET dice (cropped space): {[float(d) for d in mean_dices]}")
    print(f"now run: python 5_compute_metrics.py --pred_name swinunetr")
