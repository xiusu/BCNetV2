import torch
import torch.nn as nn
from .ops import SlimmableConv2d, SlimmableBatchNorm2d, SlimmableLinear, SlimmableInvertedResidual
from .builder import BackboneReg

# [8, 8, 96, 16, 96, 16, 96, 24, 144, 24, 144, 24, 144, 48, 288, 48, 288, 48, 288, 48, 288, 64, 432, 64, 432, 64, 648, 176, 720, 176, 720, 176, 1440, 280, 1920]  # 207M, dynamic 12
# [8, 8, 96, 16, 96, 16, 96, 24, 144, 24, 144, 24, 144, 56, 288, 56, 288, 56, 288, 56, 288, 96, 432, 96, 432, 96, 864, 240, 1440, 240, 960, 240, 1440, 480, 1920]  # 305M, dynamic 12
# [32, 16, 144, 24, 176, 24, 192, 48, 240, 48, 144, 48, 264, 88, 288, 88, 336, 88, 432, 88, 576, 144, 576, 144, 648, 144, 864, 240, 1440, 240, 1440, 240, 1440, 480, 1920]  # 505M, dynamic 12
@BackboneReg.register_module('autoslimsearchmobilenetv2')
class AutoSlimSearchMobileNetV2(nn.Module):
    def __init__(self,
                 block_setting=None,
                 last_channel_setting=None,
                 num_classes=1000,
                 input_size=224):
        super(AutoSlimSearchMobileNetV2, self).__init__()

        if block_setting is None:
            self.block_setting = [
                # n, s, expand, [out_min, out_max, step]
                [1, 1, 1, [4, 36, 4]], # [1, 16, 1, 1],
                [2, 2, 6, [8, 40, 4]], # [6, 24, 2, 2],
                [3, 2, 6, [16, 48, 4]], # [6, 32, 3, 2],
                [4, 2, 6, [32, 96, 8]], # [6, 64, 4, 2],
                [3, 1, 6, [48, 144, 12]], # [6, 96, 3, 1],
                [3, 2, 6, [80, 240, 20]], # [6, 160, 3, 2],
                [1, 1, 6, [160, 480, 40]], # [6, 320, 1, 1],
            ]
            self.last_channel_setting = [640, 1920, 160]
        else:
            self.block_setting = block_setting
            self.last_channel_setting = last_channel_setting

        self.shotcut_channel_candidates = []
        self.shotcut_channel_step = []
        self.inner_channel_candidates = []
        self.inner_channel_step = []
        self.last_channel_candidates = []
        self.last_channel_step = []
        for n, _, expand, out in self.block_setting:
            self.shotcut_channel_candidates.append(list(range(out[0], out[1]+1, out[2])))
            self.shotcut_channel_step.append(out[2])
            out_expand = [t * expand for t in out]
            for _ in range(n):
                self.inner_channel_candidates.append(list(range(out_expand[0], out_expand[1]+1, out_expand[2])))
                self.inner_channel_step.append(out_expand[2])
        self.last_channel_candidates.append(list(range(self.last_channel_setting[0], self.last_channel_setting[1]+1, self.last_channel_setting[2])))
        self.last_channel_step.append(self.last_channel_setting[2])

        self.channel_candidates = self.shotcut_channel_candidates + self.inner_channel_candidates + self.last_channel_candidates
        self.channel_step = self.shotcut_channel_step + self.inner_channel_step + self.last_channel_step
        self.current_channel_choice = []

        # stem
        output_channel = max(self.inner_channel_candidates[0])
        self.features = [nn.Sequential(
            SlimmableConv2d(3, output_channel, 3, 2, 1, bias=False),
            SlimmableBatchNorm2d(output_channel),
            nn.ReLU(inplace=True))]
        input_channel = output_channel

        for n, s, expand, out in self.block_setting:
            output_channel = out[1]
            for i in range(n):
                stride = s if i == 0 else 1
                use_shortcut = False if i == 0 else True
                self.features.append(SlimmableInvertedResidual(
                    input_channel,
                    output_channel,
                    stride,
                    expand,
                    use_shortcut=use_shortcut))
                input_channel = output_channel

        last_channel = max(self.last_channel_candidates[0])
        self.features.append(nn.Sequential(
            SlimmableConv2d(input_channel, last_channel, 1, 1, 0, bias=False),
            SlimmableBatchNorm2d(last_channel),
            nn.ReLU(inplace=True)))

        self.features = nn.Sequential(*self.features)
        self.classifier = SlimmableLinear(last_channel, num_classes)

        self.set_choice([max(candidates) for candidates in self.channel_candidates])

        # weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)


    def forward(self, x):
        x = self.features(x)
        x = nn.functional.adaptive_avg_pool2d(x, (1, 1))
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

    def set_choice(self, choice_list):
        for i, choice in enumerate(choice_list):
            assert choice in self.channel_candidates[i]

        self.current_channel_choice = choice_list

        shortcut_choice = choice_list[0:len(self.shotcut_channel_candidates)]
        choice_list = choice_list[len(shortcut_choice):]
        inner_choice = choice_list[0:len(self.inner_channel_candidates)]
        choice_list = choice_list[len(inner_choice):]
        last_choice = choice_list[0:len(self.last_channel_candidates)]

        self.features[0][0].width = inner_choice[0]
        stage_idx = 0
        block_idx = 0
        for n, _, _, _ in self.block_setting:
            for _ in range(n):
                self.features[block_idx+1].set_width(inner_choice[block_idx], shortcut_choice[stage_idx])
                block_idx += 1
            stage_idx += 1
        self.features[-1][0].width = last_choice[0]
