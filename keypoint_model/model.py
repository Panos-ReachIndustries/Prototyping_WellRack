"""
Vial Keypoint Detection Model.

Simple Baselines architecture (Xiao et al., 2018):
  ResNet backbone → 3 deconv layers → 1×1 conv → heatmap

Input:  (B, 3, H, W)  e.g. (B, 3, 512, 512)
Output: (B, 1, H/4, W/4)  e.g. (B, 1, 128, 128) single-channel heatmap
"""
import torch
import torch.nn as nn
import torchvision.models as models

from .config import ModelConfig


_RESNET_FACTORY = {
    "resnet18": (models.resnet18, models.ResNet18_Weights.DEFAULT, 512),
    "resnet34": (models.resnet34, models.ResNet34_Weights.DEFAULT, 512),
    "resnet50": (models.resnet50, models.ResNet50_Weights.DEFAULT, 2048),
}


class DeconvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=4, stride=2, padding=1):
        super().__init__()
        self.block = nn.Sequential(
            nn.ConvTranspose2d(in_ch, out_ch, kernel_size, stride, padding, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class VialKeypointNet(nn.Module):
    """
    Heatmap regression network for vial centre detection.

    Architecture:
        ResNet backbone (layer1..layer4) → 3× DeconvBlock → 1×1 Conv → sigmoid
        Output stride = 4  (H_out = H_in / 4, W_out = W_in / 4)
    """

    def __init__(self, cfg: ModelConfig | None = None):
        super().__init__()
        if cfg is None:
            cfg = ModelConfig()

        if cfg.backbone not in _RESNET_FACTORY:
            raise ValueError(f"Unsupported backbone: {cfg.backbone}. Choose from {list(_RESNET_FACTORY)}")

        factory_fn, weights, backbone_out_ch = _RESNET_FACTORY[cfg.backbone]
        resnet = factory_fn(weights=weights if cfg.pretrained else None)

        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4

        assert len(cfg.num_deconv_filters) == cfg.num_deconv_layers
        assert len(cfg.num_deconv_kernels) == cfg.num_deconv_layers

        deconv_layers = []
        in_ch = backbone_out_ch
        for i in range(cfg.num_deconv_layers):
            out_ch = cfg.num_deconv_filters[i]
            k = cfg.num_deconv_kernels[i]
            pad = 1 if k == 4 else 0
            deconv_layers.append(DeconvBlock(in_ch, out_ch, kernel_size=k, stride=2, padding=pad))
            in_ch = out_ch

        self.deconv = nn.Sequential(*deconv_layers)
        self.head = nn.Conv2d(in_ch, cfg.num_channels, kernel_size=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.deconv(x)
        x = self.head(x)
        return x
