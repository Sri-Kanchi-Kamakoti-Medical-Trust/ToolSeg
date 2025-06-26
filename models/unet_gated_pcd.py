import torch
import torch.nn as nn
import torch.nn.functional as F

from abc import ABCMeta, abstractmethod
from collections import OrderedDict

from layers import DoubleConv, Down, Up, OutConv
from unet_pcd import UpscalePCD, PCDBase


class GatedPCDBlock(PCDBase):
    def __init__(self, in_channels: int, out_channels: int, n_phases: int, bn_momentum: float = 0.1, stride: int = 2, activation: str = "linear", scale: bool = True, shift: bool = True) -> None:
        super().__init__(in_channels, out_channels, n_phases, bn_momentum, stride, activation, scale, shift)
        self.gate_alpha = nn.Linear(n_phases, self.pcd_dims)
        self.feat_alpha = nn.Conv2d(in_channels, self.pcd_dims, kernel_size=5, stride=1, padding=2)

    def rescale_features(self, feature_map, x_aux):
        beta = self.phase_transform_beta(x_aux)
        gamma = self.phase_transform_gamma(x_aux)
        gate_alpha = self.gate_alpha(x_aux)
        feat_alpha = self.feat_alpha(feature_map)
        beta = beta.view(*beta.size(), 1, 1).expand_as(feature_map)
        gamma = gamma.view(*gamma.size(), 1, 1).expand_as(feature_map)
        gate_alpha = gate_alpha.view(*gate_alpha.size(), 1, 1).expand_as(feature_map)
        gate_alpha = (gate_alpha + feat_alpha) / 2
        feature_map_transformed = (gamma * feature_map) + beta
        feature_map = gate_alpha * feature_map_transformed + (1 - gate_alpha) * feature_map
        if self.scale_activation is not None:
            feature_map = self.scale_activation(feature_map)
        return feature_map

    def forward(self, feature_map, x_aux):
        out = self.rescale_features(feature_map, x_aux)
        return out
    

class UNetGatedPCD(nn.Module):
    def __init__(self, n_channels, n_classes, n_phases, bilinear=False):
        super(UNetGatedPCD, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = (DoubleConv(n_channels, 64))
        self.down1 = (Down(64, 128))
        self.down2 = (Down(128, 256))
        self.down3 = (Down(256, 512))
        factor = 2 if bilinear else 1
        self.down4 = (Down(512, 1024 // factor))

        self.pcd_dc1 = GatedPCDBlock(in_channels=512, out_channels=512, n_phases=n_phases)
        self.pcd_dc2 = GatedPCDBlock(in_channels=256, out_channels=256, n_phases=n_phases)
        self.pcd_dc3 = GatedPCDBlock(in_channels=128, out_channels=128, n_phases=n_phases)
        self.pcd_dc4 = GatedPCDBlock(in_channels=64, out_channels=64, n_phases=n_phases)

        self.up1 = (UpscalePCD(1024, 512 // factor, bilinear))
        self.up2 = (UpscalePCD(512, 256 // factor, bilinear))
        self.up3 = (UpscalePCD(256, 128 // factor, bilinear))
        self.up4 = (UpscalePCD(128, 64, bilinear))
        self.outc = (OutConv(64, n_classes))

        print("UNetGatedPCD model created")

    def forward(self, x, phase):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4, self.pcd_dc1, phase)
        x = self.up2(x, x3, self.pcd_dc2, phase)
        x = self.up3(x, x2, self.pcd_dc3, phase)
        x = self.up4(x, x1, self.pcd_dc4, phase)
        logits = self.outc(x)
        return logits

if __name__ == "__main__":
    model = UNetGatedPCD(3, 14, 20)

    image = torch.randn(16, 3, 256, 256)
    phase = torch.randn(16, 20)
    print(image.shape, phase.shape)

    output = model(image, phase)
    print(output.shape)