"""Official SAHCC predictor with an offline, feature-only VGG16-BN loader."""
from dataclasses import asdict, dataclass

from .encoder import VGG16BN
from .sahcc import P2PNet


@dataclass
class ModelConfig:
    row: int = 2
    line: int = 2
    head_hidden_dim: int = 256
    decoder_out_channels: int = 256
    anchor_pyramid_level: int = 3
    offset_scale: float = 100.
    directional_detail: bool = False

    def to_dict(self):
        result = asdict(self)
        if not self.directional_detail:
            result.pop("directional_detail")  # Retain legacy checkpoint config.
        return result


class SAHCC(P2PNet):
    def __init__(self, config=None, pretrained=None):
        config = config or ModelConfig()
        if (config.row < 1 or config.line < 1 or config.anchor_pyramid_level != 3 or
                min(config.head_hidden_dim, config.decoder_out_channels) < 1):
            raise ValueError('This decoder has stride 8; anchor pyramid level must be 3')
        super().__init__(VGG16BN(pretrained), **config.to_dict())
        self.config = config

    def forward(self, images, targets=None):
        if images.ndim != 4 or images.shape[1] != 3:
            raise ValueError('Expected Bx3xHxW RGB images')
        if min(images.shape[-2:]) < 32 or any(n % 16 for n in images.shape[-2:]):
            raise ValueError('Input dimensions must be >=32 and divisible by 16')
        return super().forward(images)
