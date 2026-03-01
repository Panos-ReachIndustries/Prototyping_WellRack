from dataclasses import dataclass, field
from typing import Tuple


@dataclass
class ModelConfig:
    backbone: str = "resnet18"
    pretrained: bool = True
    num_deconv_layers: int = 3
    num_deconv_filters: Tuple[int, ...] = (256, 128, 64)
    num_deconv_kernels: Tuple[int, ...] = (4, 4, 4)
    num_channels: int = 1


@dataclass
class TrainConfig:
    input_size: Tuple[int, int] = (512, 512)
    heatmap_stride: int = 4
    heatmap_sigma: float = 2.5
    batch_size: int = 4
    num_workers: int = 2
    lr: float = 1e-3
    weight_decay: float = 1e-4
    epochs: int = 200
    lr_step: Tuple[int, ...] = (90, 140, 180)
    lr_factor: float = 0.1
    augment: bool = True
    val_split: float = 0.15
    seed: int = 42
    save_dir: str = "checkpoints"
    log_interval: int = 10
    annotations_path: str = "assets/testing_images/vial_keypoints.json"
    images_dir: str = "assets/testing_images"
