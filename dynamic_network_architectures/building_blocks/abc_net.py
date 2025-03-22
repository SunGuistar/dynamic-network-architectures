import torch
import torch.nn as nn
import torch.nn.functional as F

class ASPP(nn.Module):
    def __init__(self, in_channels, out_channels, dilation_rates=(6, 12, 18)):
        super(ASPP, self).__init__()

        self.conv1x1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0, bias=False)
        self.conv3x3_1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=dilation_rates[0],
                                   dilation=dilation_rates[0], bias=False)
        self.conv3x3_2 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=dilation_rates[1],
                                   dilation=dilation_rates[1], bias=False)
        self.conv3x3_3 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=dilation_rates[2],
                                   dilation=dilation_rates[2], bias=False)

        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.global_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)

        # 输出通道数应与 `out_channels` 保持一致
        self.final_conv = nn.Conv2d(out_channels * 5, out_channels, kernel_size=1, bias=False)

    def forward(self, x):
        # 多尺度空洞卷积
        aspp1x1 = self.conv1x1(x)
        aspp3x3_1 = self.conv3x3_1(x)
        aspp3x3_2 = self.conv3x3_2(x)
        aspp3x3_3 = self.conv3x3_3(x)

        # 全局平均池化
        global_feature = self.global_avg_pool(x)
        global_feature = self.global_conv(global_feature)
        global_feature = F.interpolate(global_feature, size=x.shape[2:], mode="bilinear", align_corners=True)

        # 拼接
        out = torch.cat([aspp1x1, aspp3x3_1, aspp3x3_2, aspp3x3_3, global_feature], dim=1)
        out = self.final_conv(out)

        return out