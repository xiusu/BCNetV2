import torch
import torch.nn as nn
from .choice import Choice
from .list_choice import ListChoice

from .ops import Identity, InvertedResidual
from .builder import BackboneReg

def create_choice_block(inp, oup, stride, index=None):
    op_list = [InvertedResidual(inp, oup, stride, expand_ratio=3, kernel_size=3),
               InvertedResidual(inp, oup, stride, expand_ratio=3, kernel_size=5),
               InvertedResidual(inp, oup, stride, expand_ratio=3, kernel_size=7),
               InvertedResidual(inp, oup, stride, expand_ratio=6, kernel_size=3),
               InvertedResidual(inp, oup, stride, expand_ratio=6, kernel_size=5),
               InvertedResidual(inp, oup, stride, expand_ratio=6, kernel_size=7),
               Identity(inp, oup, stride)]
    if index is None:
        return ListChoice(op_list)
    else:
        return op_list[index]

@BackboneReg.register_module('mobilenetv2')
class MobileNetV2(nn.Module):
    def __init__(self,
                 num_classes=1000,
                 input_size=224,
                 width_mult=1.,
                 architect=None):
        super(MobileNetV2, self).__init__()
        first_channel = 32
        last_channel = 1280
        interverted_residual_setting = [
            [32, 4, 2],
            [40, 4, 2],
            [80, 4, 2],
            [96, 4, 1],
            [192, 4, 2],
            [320, 1, 1],
        ]

        assert input_size % 32 == 0
        self.first_channel = int(first_channel * width_mult)
        self.last_channel = int(last_channel * width_mult) if width_mult > 1.0 else last_channel

        # stem
        output_channel = self.first_channel
        self.features = [nn.Sequential(
            nn.Conv2d(3, output_channel, 3, 2, 1, bias=False),
            nn.BatchNorm2d(output_channel),
            nn.ReLU(inplace=False))]
        input_channel = output_channel
        output_channel = int(16 * width_mult)
        self.features.append(InvertedResidual(input_channel, output_channel, 1, expand_ratio=1))
        input_channel = output_channel

        cnt = 0
        for channel, n, stride in interverted_residual_setting:
            output_channel = int(channel * width_mult)
            for i in range(n):
                if architect is not None:
                    self.features.append(create_choice_block(
                        input_channel, output_channel, stride if i == 0 else 1, architect[cnt]))
                else:
                    self.features.append(create_choice_block(
                        input_channel, output_channel, stride if i == 0 else 1))
                cnt += 1
                input_channel = output_channel

        self.features.append(nn.Sequential(
            nn.Conv2d(input_channel, self.last_channel, 1, 1, 0, bias=False),
            nn.BatchNorm2d(self.last_channel),
            nn.ReLU(inplace=False)))

        self.features = nn.Sequential(*self.features)

        self.classifier = nn.Linear(self.last_channel, num_classes)

        self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = nn.functional.adaptive_avg_pool2d(x, (1, 1))
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
