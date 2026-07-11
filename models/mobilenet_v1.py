"""
MobileNetV1 implementation from scratch (depthwise separable convolutions)
بدون استفاده از مدل‌های آماده (torchvision)

تابع mobilenet_v1(...) برای ساخت مدل با پارامترهای استاندارد قرار داده شده است.
"""

from typing import List

import torch
import torch.nn as nn


def _make_divisible(v: float, divisor: int = 8, min_value: int = None) -> int:
    """از نسخه‌ی اصلی MobileNet: اطمینان حاصل می‌کند که تعداد فیچرها به مضرب divisor گرد شود."""
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


def conv_bn(in_channels: int, out_channels: int, stride: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, stride, 1, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
    )


def conv_dw(in_channels: int, out_channels: int, stride: int) -> nn.Sequential:
    """Depthwise separable conv: depthwise (groups=in_channels) + pointwise"""
    return nn.Sequential(
        # depthwise
        nn.Conv2d(in_channels, in_channels, 3, stride, 1, groups=in_channels, bias=False),
        nn.BatchNorm2d(in_channels),
        nn.ReLU(inplace=True),
        # pointwise
        nn.Conv2d(in_channels, out_channels, 1, 1, 0, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
    )


class MobileNetV1(nn.Module):
    """MobileNetV1

    معماری مطابق با مقاله‌ی اولیه: https://arxiv.org/abs/1704.04861

    پارامترها:
      - num_classes: تعداد کلاس‌ها برای لایه‌ی خروجی (default=1000)
      - width_mult: ضریب پهنای شبکه برای کوچک‌سازی/بزرگ‌سازی تعداد کانال‌ها
    """

    def __init__(self, num_classes: int = 1000, width_mult: float = 1.0):
        super().__init__()
        input_channel = 32
        last_channel = 1024

        input_channel = _make_divisible(input_channel * width_mult)
        self.features = []

        # initial conv
        self.features.append(conv_bn(3, input_channel, stride=2))

        # settings: (out_channels, stride) for each depthwise block
        cfg: List[tuple] = [
            (64, 1),
            (128, 2),
            (128, 1),
            (256, 2),
            (256, 1),
            (512, 2),
            (512, 1),
            (512, 1),
            (512, 1),
            (512, 1),
            (512, 1),
            (1024, 2),
            (1024, 1),
        ]

        for out_c, s in cfg:
            output_channel = _make_divisible(out_c * width_mult)
            self.features.append(conv_dw(input_channel, output_channel, stride=s))
            input_channel = output_channel

        # classifier
        self.features = nn.Sequential(*self.features)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(input_channel, num_classes)

        self._initialize_weights()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

    def _initialize_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)


def mobilenet_v1(num_classes: int = 1000, width_mult: float = 1.0) -> MobileNetV1:
    """factory function"""
    return MobileNetV1(num_classes=num_classes, width_mult=width_mult)


__all__ = ["MobileNetV1", "mobilenet_v1"]
