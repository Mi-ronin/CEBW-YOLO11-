import torch
import torch.nn as nn
import torch.nn.functional as F



# -----------------------------
class Conv(nn.Module):
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU() if act else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


def autopad(k, p=None):
    if p is None:
        p = k // 2
    return p


# -----------------------------
# DWR模块（核心创新点）
# -----------------------------
class DWR(nn.Module):
    """
    Dilated Wide Residual Block
    """

    def __init__(self, c):
        super().__init__()

        self.conv1 = Conv(c, c, 3, 1)

        # 多膨胀卷积
        self.dilated1 = nn.Conv2d(c, c, 3, padding=1, dilation=1, bias=False)
        self.dilated2 = nn.Conv2d(c, c, 3, padding=2, dilation=2, bias=False)
        self.dilated3 = nn.Conv2d(c, c, 3, padding=3, dilation=3, bias=False)

        self.bn = nn.BatchNorm2d(c)
        self.act = nn.SiLU()

        self.conv2 = Conv(c, c, 1, 1)

    def forward(self, x):
        identity = x

        x = self.conv1(x)

        # 多尺度特征融合
        d1 = self.dilated1(x)
        d2 = self.dilated2(x)
        d3 = self.dilated3(x)

        x = d1 + d2 + d3
        x = self.act(self.bn(x))

        x = self.conv2(x)

        # 残差连接
        return x + identity


# -----------------------------
# Bottleneck with DWR
# -----------------------------
class Bottleneck_DWR(nn.Module):
    def __init__(self, c1, c2, shortcut=True):
        super().__init__()
        self.cv1 = Conv(c1, c2, 1, 1)
        self.dwr = DWR(c2)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        y = self.dwr(self.cv1(x))
        return x + y if self.add else y


# -----------------------------
# C3k2-DWR 模块
# -----------------------------
class C3k2_DWR(nn.Module):
    """
    C3 with k=2 + DWR
    """

    def __init__(self, c1, c2, n=2, shortcut=True):
        super().__init__()
        c_ = int(c2 * 0.5)

        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(2 * c_, c2, 1)

        # k2：两个DWR bottleneck
        self.m = nn.Sequential(*[
            Bottleneck_DWR(c_, c_, shortcut) for _ in range(n)
        ])

    def forward(self, x):
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), dim=1))