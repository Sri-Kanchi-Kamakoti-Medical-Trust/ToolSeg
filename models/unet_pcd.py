import torch
import torch.nn as nn
import torch.nn.functional as F

from abc import ABCMeta, abstractmethod
from collections import OrderedDict

from layers import DoubleConv, Down, Up, OutConv


class UpscalePCD(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2, phase_block, phase):
        x1 = self.up(x1)
        x1 = phase_block(x1, phase)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class PCDBase(nn.Module, metaclass=ABCMeta):
    """Absract base class for models that are related to FiLM of Perez et al"""

    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            n_phases: int,
            bn_momentum: float = 0.1,
            stride: int = 2,
            activation: str = "linear",
            scale: bool = True,
            shift: bool = True,
    ) -> None:

        super().__init__()

        # sanity checks
        if (not isinstance(scale, bool) or not isinstance(shift, bool)) or (not scale and not shift):
            raise ValueError(
                f"scale and shift must be of type bool:\n    -> scale value: {scale}, "
                "scale type {type(scale)}\n    -> shift value: {shift}, shift type: {type(shift)}"
            )
        # ResBlock
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        # location decoding
        self.pcd_dims = in_channels
        self.phase_transform_beta = nn.Linear(n_phases, self.pcd_dims)
        self.phase_transform_gamma = nn.Linear(n_phases, self.pcd_dims)
        if activation == "sigmoid":
            self.scale_activation = nn.Sigmoid()
        elif activation == "tanh":
            self.scale_activation = nn.Tanh()
        elif activation == "linear":
            self.scale_activation = None

    # @abstractmethod
    def rescale_features(self, feature_map, x_aux):
        """method to recalibrate feature map x"""
        beta = self.phase_transform_beta(x_aux)
        gamma = self.phase_transform_gamma(x_aux)
        beta = beta.view(*beta.size(), 1, 1).expand_as(feature_map)
        gamma = gamma.view(*gamma.size(), 1, 1).expand_as(feature_map)
        feature_map = (gamma * feature_map) + beta
        if self.scale_activation is not None:
            feature_map = self.scale_activation(feature_map)
        return feature_map



    def forward(self, feature_map, x_aux):
        out = self.rescale_features(feature_map, x_aux)
        return out

    

class UNetPCD(nn.Module):
    def __init__(self, n_channels, n_classes, n_phases, bilinear=False):
        super(UNetPCD, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = (DoubleConv(n_channels, 64))
        self.down1 = (Down(64, 128))
        self.down2 = (Down(128, 256))
        self.down3 = (Down(256, 512))
        factor = 2 if bilinear else 1
        self.down4 = (Down(512, 1024 // factor))

        self.pcd_dc1 = PCDBase(in_channels=512, out_channels=512, n_phases=n_phases)
        self.pcd_dc2 = PCDBase(in_channels=256, out_channels=256, n_phases=n_phases)
        self.pcd_dc3 = PCDBase(in_channels=128, out_channels=128, n_phases=n_phases)
        self.pcd_dc4 = PCDBase(in_channels=64, out_channels=64, n_phases=n_phases)

        self.up1 = (UpscalePCD(1024, 512 // factor, bilinear))
        self.up2 = (UpscalePCD(512, 256 // factor, bilinear))
        self.up3 = (UpscalePCD(256, 128 // factor, bilinear))
        self.up4 = (UpscalePCD(128, 64, bilinear))
        self.outc = (OutConv(64, n_classes))

        print("UNetPCD Model Created")

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
    model = UNetPCD(3, 14, 20)

    image = torch.randn(16, 3, 256, 256)
    phase = torch.randn(16, 20)
    print(image.shape, phase.shape)

    output = model(image, phase)
    print(output.shape)