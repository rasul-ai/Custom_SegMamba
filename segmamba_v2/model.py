"""SegMamba with deep-supervision heads. The backbone is the unchanged model_segmamba.SegMamba."""
import torch.nn as nn
from monai.networks.blocks.dynunet_block import UnetOutBlock

from model_segmamba.segmamba import SegMamba


class SegMambaDS(SegMamba):
    def __init__(self, deep_supervision=True, **kwargs):
        super().__init__(**kwargs)
        self.deep_supervision = deep_supervision
        # 1x1x1 heads on the decoder outputs at 1/2, 1/4 and 1/8 resolution (96, 192, 384 channels).
        self.ds_heads = nn.ModuleList([
            UnetOutBlock(spatial_dims=self.spatial_dims, in_channels=c, out_channels=self.out_chans)
            for c in self.feat_size[1:]
        ])

    def forward(self, x_in):
        # Same as SegMamba.forward, but keeps the intermediate decoder outputs.
        outs = self.vit(x_in)
        enc1 = self.encoder1(x_in)
        enc2 = self.encoder2(outs[0])
        enc3 = self.encoder3(outs[1])
        enc4 = self.encoder4(outs[2])
        enc_hidden = self.encoder5(outs[3])
        dec3 = self.decoder5(enc_hidden, enc4)  # 1/8
        dec2 = self.decoder4(dec3, enc3)        # 1/4
        dec1 = self.decoder3(dec2, enc2)        # 1/2
        dec0 = self.decoder2(dec1, enc1)        # full resolution
        out = self.out(self.decoder1(dec0))

        if self.training and self.deep_supervision:
            # full resolution first; inference (eval mode) only returns `out`
            return [out] + [head(d) for head, d in zip(self.ds_heads, (dec1, dec2, dec3))]
        return out
