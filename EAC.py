import torch
import torch.nn as nn
import torch.nn.functional as F


# -----------------------------
# 基础Conv
# -----------------------------
class Conv(nn.Module):
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, self.autopad(k, p), groups=g, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU() if act else nn.Identity()

    def autopad(self, k, p=None):
        return k // 2 if p is None else p

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


# -----------------------------
# 通道注意力（轻量版SE）
# -----------------------------
class ChannelAttention(nn.Module):
    def __init__(self, c, r=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(c, c // r, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(c // r, c, 1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        return x * self.fc(self.avg_pool(x))


# -----------------------------
# 空间注意力（增强边缘敏感）
# -----------------------------
class SpatialAttention(nn.Module):
    def __init__(self, k=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, k, padding=k // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg = torch.mean(x, dim=1, keepdim=True)
        max, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg, max], dim=1)
        return self.sigmoid(self.conv(x))


# -----------------------------
# 边缘增强（Sobel算子）
# -----------------------------
class EdgeEnhance(nn.Module):
    def __init__(self, c):
        super().__init__()

        sobel_x = torch.tensor([[-1, 0, 1],
                                [-2, 0, 2],
                                [-1, 0, 1]], dtype=torch.float32).unsqueeze(0).unsqueeze(0)

        sobel_y = torch.tensor([[-1, -2, -1],
                                [0,  0,  0],
                                [1,  2,  1]], dtype=torch.float32).unsqueeze(0).unsqueeze(0)

        self.weight_x = nn.Parameter(sobel_x.repeat(c, 1, 1, 1), requires_grad=False)
        self.weight_y = nn.Parameter(sobel_y.repeat(c, 1, 1, 1), requires_grad=False)

        self.groups = c

    def forward(self, x):
        edge_x = F.conv2d(x, self.weight_x, padding=1, groups=self.groups)
        edge_y = F.conv2d(x, self.weight_y, padding=1, groups=self.groups)
        edge = torch.abs(edge_x) + torch.abs(edge_y)
        return edge


# -----------------------------
# EAC模块（核心）
# -----------------------------
class EAC(nn.Module):
    def __init__(self, c):
        super().__init__()

        self.conv1 = Conv(c, c, 3, 1)

        # 边缘增强
        self.edge = EdgeEnhance(c)

        # 注意力
        self.ca = ChannelAttention(c)
        self.sa = SpatialAttention()

        self.conv2 = Conv(c, c, 1, 1)

    def forward(self, x):
        identity = x

        x = self.conv1(x)

        # 边缘信息
        edge = self.edge(x)

        # 融合（关键创新点）
        x = x + edge

        # 注意力校准
        x = self.ca(x)
        x = x * self.sa(x)

        x = self.conv2(x)

        return x + identity