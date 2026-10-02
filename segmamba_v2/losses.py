"""Region-based BCE + soft Dice loss with deep supervision (nnU-Net style)."""
import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceBCELoss(nn.Module):
    """BCE-with-logits + (1 - soft Dice) over the sigmoid region channels (TC, WT, ET)."""

    def __init__(self, batch_dice=True, smooth=1e-5):
        super().__init__()
        self.batch_dice = batch_dice
        self.smooth = smooth

    def forward(self, logits, target):
        logits = logits.float()  # keep the loss in fp32 under autocast
        bce = F.binary_cross_entropy_with_logits(logits, target)

        probs = torch.sigmoid(logits)
        dims = (0, 2, 3, 4) if self.batch_dice else (2, 3, 4)
        intersect = (probs * target).sum(dims)
        denom = probs.sum(dims) + target.sum(dims)
        dice = (2 * intersect + self.smooth) / (denom + self.smooth).clamp_min(1e-8)
        return bce + (1 - dice.mean())


class DeepSupervisionLoss(nn.Module):
    """Weights 1, 1/2, 1/4, ... (normalised); the target is downsampled to each output's size."""

    def __init__(self, loss):
        super().__init__()
        self.loss = loss

    def forward(self, outputs, target):
        if not isinstance(outputs, (list, tuple)):
            return self.loss(outputs, target)

        weights = [1 / 2 ** i for i in range(len(outputs))]
        total = sum(weights)
        loss = 0.0
        for w, out in zip(weights, outputs):
            t = target if out.shape[2:] == target.shape[2:] else F.interpolate(target, size=out.shape[2:], mode="nearest")
            loss = loss + (w / total) * self.loss(out, t)
        return loss
