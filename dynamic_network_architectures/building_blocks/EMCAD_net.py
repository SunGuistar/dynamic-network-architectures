import torch
import torch.nn as nn
from functools import partial

import math
from timm.layers import trunc_normal_tf_
from timm.models import named_apply

def channel_shuffle(x, groups):
    batchsize, num_channels, height, width = x.data.size()
    channels_per_group = num_channels // groups
    # reshape
    x = x.view(batchsize, groups,
               channels_per_group, height, width)
    x = torch.transpose(x, 1, 2).contiguous()
    # flatten
    x = x.view(batchsize, -1, height, width)
    return x

def _init_weights(module, name, scheme=''):
    if isinstance(module, nn.Conv2d) or isinstance(module, nn.Conv3d):
        if scheme == 'normal':
            nn.init.normal_(module.weight, std=.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif scheme == 'trunc_normal':
            trunc_normal_tf_(module.weight, std=.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif scheme == 'xavier_normal':
            nn.init.xavier_normal_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif scheme == 'kaiming_normal':
            nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        else:
            # efficientnet like
            fan_out = module.kernel_size[0] * module.kernel_size[1] * module.out_channels
            fan_out //= module.groups
            nn.init.normal_(module.weight, 0, math.sqrt(2.0 / fan_out))
            if module.bias is not None:
                nn.init.zeros_(module.bias)
    elif isinstance(module, nn.BatchNorm2d) or isinstance(module, nn.BatchNorm3d):
        nn.init.constant_(module.weight, 1)
        nn.init.constant_(module.bias, 0)
    elif isinstance(module, nn.LayerNorm):
        nn.init.constant_(module.weight, 1)
        nn.init.constant_(module.bias, 0)

def act_layer(act, inplace=False, neg_slope=0.2, n_prelu=1):
    # activation layer
    act = act.lower()
    if act == 'relu':
        layer = nn.ReLU(inplace)
    elif act == 'relu6':
        layer = nn.ReLU6(inplace)
    elif act == 'leakyrelu':
        layer = nn.LeakyReLU(neg_slope, inplace)
    elif act == 'prelu':
        layer = nn.PReLU(num_parameters=n_prelu, init=neg_slope)
    elif act == 'gelu':
        layer = nn.GELU()
    elif act == 'hswish':
        layer = nn.Hardswish(inplace)
    else:
        raise NotImplementedError('activation layer [%s] is not found' % act)
    return layer

# 跳跃连接
class LGAG_Modified(nn.Module):
    def __init__(self, F_g, F_l, F_int, kernel_size=3, groups=1, activation='relu'):
        super(LGAG_Modified, self).__init__()

        if kernel_size == 1:
            groups = 1
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=kernel_size, stride=1, padding=kernel_size // 2, groups=groups, bias=True),
            nn.BatchNorm2d(F_int)
        )
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=kernel_size, stride=1, padding=kernel_size // 2, groups=groups, bias=True),
            nn.BatchNorm2d(F_int)
        )
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        self.activation = act_layer(activation, inplace=True)
        self.init_weights('normal')

    def init_weights(self, scheme=''):
        named_apply(partial(_init_weights, scheme=scheme), self)

    def forward(self, g, x):
        # print('**************************进入LGAG*******************************////')
        g1 = self.W_g(g) #上采样
        x1 = self.W_x(x) #跳跃连接
        psi = self.activation(g1 + x1)
        psi = self.psi(psi)

        x_att = x * psi  # 对跳跃连接特征进行注意力加权
        return torch.cat((g, x_att), 1)  # 保持与 UNetResDecoder 相同的通道数

# 上采样模块
class EUCB(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, activation='relu'):
        super(EUCB, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        # 先进行2倍上采样
        self.up_dwc = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),  # 保持与转置卷积的上采样一致
            nn.Conv2d(self.in_channels, self.in_channels, kernel_size=kernel_size, stride=stride,
                      padding=kernel_size // 2, groups=self.in_channels, bias=False),  # 深度可分离卷积
            nn.BatchNorm2d(self.in_channels),
            act_layer(activation, inplace=True)
        )

        # 1x1 Pointwise Convolution，调整到目标通道数
        self.pwc = nn.Sequential(
            nn.Conv2d(self.in_channels, self.out_channels, kernel_size=1, stride=1, padding=0, bias=True)
        )

    def forward(self, x):
        x = self.up_dwc(x)  # 上采样 + 深度可分离卷积
        x = channel_shuffle(x, self.in_channels)  # 进行通道混合（可选）
        x = self.pwc(x)  # 1x1 卷积调整通道数
        return x


#   Channel attention block (CAB)
class CAB(nn.Module):
    def __init__(self, ratio=16, activation='relu'):
        super(CAB, self).__init__()

        self.ratio = ratio
        self.activation = act_layer(activation, inplace=True)
        self.sigmoid = nn.Sigmoid()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        # `fc1` 和 `fc2` 不能在 `__init__` 里固定，而是在 `forward` 里动态定义
        self.fc1 = None
        self.fc2 = None

    def forward(self, x):
        in_channels = x.shape[1]  # 获取输入通道数
        reduced_channels = max(1, in_channels // self.ratio)  # 确保最小通道数不小于 1

        # 如果 `fc1` 和 `fc2` 还未定义，或者输入通道数发生变化，则动态调整
        if self.fc1 is None or self.fc1.in_channels != in_channels:
            self.fc1 = nn.Conv2d(in_channels, reduced_channels, 1, bias=False).to(x.device)
            self.fc2 = nn.Conv2d(reduced_channels, in_channels, 1, bias=False).to(x.device)

        avg_out = self.fc2(self.activation(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.activation(self.fc1(self.max_pool(x))))

        out = avg_out + max_out
        return self.sigmoid(out)



#   Spatial attention block (SAB)
class SAB(nn.Module):
    def __init__(self, kernel_size=7):
        super(SAB, self).__init__()

        assert kernel_size in (3, 7, 11), "Kernel size must be 3, 7, or 11"
        self.kernel_size = kernel_size
        self.conv = None  # 延迟初始化
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        in_channels = x.shape[1]  # 获取输入通道数
        padding = self.kernel_size // 2

        # 如果 `conv` 还未定义，或者输入通道数发生变化，则动态调整
        if self.conv is None or self.conv.in_channels != 2:
            self.conv = nn.Conv2d(2, 1, self.kernel_size, padding=padding, bias=False).to(x.device)

        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv(x)
        return self.sigmoid(x)
