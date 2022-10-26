import torch
import torch.nn as nn
from .ops import InvertedResidual, SlimmableInvertedResidual
from .builder import BackboneReg

# [8, 8, 96, 16, 96, 16, 96, 24, 144, 24, 144, 24, 144, 48, 288, 48, 288, 48, 288, 48, 288, 64, 432, 64, 432, 64, 648, 176, 720, 176, 720, 176, 1440, 280, 1920]  # 207M, dynamic 12
# [8, 8, 96, 16, 96, 16, 96, 24, 144, 24, 144, 24, 144, 56, 288, 56, 288, 56, 288, 56, 288, 96, 432, 96, 432, 96, 864, 240, 1440, 240, 960, 240, 1440, 480, 1920]  # 305M, dynamic 12
# [32, 16, 144, 24, 176, 24, 192, 48, 240, 48, 144, 48, 264, 88, 288, 88, 336, 88, 432, 88, 576, 144, 576, 144, 648, 144, 864, 240, 1440, 240, 1440, 240, 1440, 480, 1920]  # 505M, dynamic 12
@BackboneReg.register_module('autoslimmobilenetv2')
class AutoSlimMobileNetV2(nn.Module):
    def __init__(self,
                 architect,
                 num_classes=1000,
                 input_size=224):
        super(AutoSlimMobileNetV2, self).__init__()
        first_channel = architect[0]
        last_channel = architect[-1]
        strides = [1, 2, 1, 2, 1, 1, 2, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1]
        shortcut = [False, False, True, False, True, True, False, True, True, True, False, True, True, False, True, True, False]

        assert len(strides) * 2 + 1 == len(architect)
        assert input_size % 32 == 0
        # stem
        output_channel = first_channel
        self.features = [nn.Sequential(
            nn.Conv2d(3, output_channel, 3, 2, 1, bias=False),
            nn.BatchNorm2d(output_channel),
            nn.ReLU(inplace=True))]
        input_channel = output_channel

        for i in range(0, len(architect) - 1, 2):
            mid_channel = architect[i]
            output_channel = architect[i+1]
            self.features.append(InvertedResidual(
                input_channel,
                output_channel,
                strides[i//2],
                mid_channel/input_channel,
                use_shortcut=shortcut[i//2]))
            input_channel = output_channel

        self.features.append(nn.Sequential(
            nn.Conv2d(input_channel, last_channel, 1, 1, 0, bias=False),
            nn.BatchNorm2d(last_channel),
            nn.ReLU(inplace=True)))

        self.features = nn.Sequential(*self.features)
        self.classifier = nn.Linear(last_channel, num_classes)
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
