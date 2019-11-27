import torch.nn as nn

class Conv2_5d(nn.Conv3d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode=None,
                        unsqueeze_axis=-3):
        self.unsqueeze_axis = unsqueeze_axis

        unsqueeze_axis += 3
        kernel_size = [kernel_size, kernel_size]
        padding = [padding, padding]
        kernel_size.insert(unsqueeze_axis, 1)
        padding.insert(unsqueeze_axis, 0)

        super().__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)