import torch
import torch.nn as nn
import torch.nn.functional as F


# -----------------------------
# 深度可分离卷积
# -----------------------------
class SeparableConv(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.depthwise = nn.Conv2d(c, c, 3, padding=1, groups=c, bias=False)
        self.pointwise = nn.Conv2d(c, c, 1, bias=False)
        self.bn = nn.BatchNorm2d(c)
        self.act = nn.SiLU()

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return self.act(self.bn(x))


# -----------------------------
# 加权融合
# -----------------------------
class FastWeightedFusion(nn.Module):
    def __init__(self, n_inputs, eps=1e-4):
        super().__init__()
        self.w = nn.Parameter(torch.ones(n_inputs))
        self.eps = eps

    def forward(self, inputs):
        w = F.relu(self.w)
        weight = w / (torch.sum(w) + self.eps)

        out = 0
        for i in range(len(inputs)):
            out = out + weight[i] * inputs[i]
        return out


# -----------------------------
# 单层 BiFPN
# -----------------------------
class BiFPN_Layer(nn.Module):
    def __init__(self, c):
        super().__init__()

        # 上采样 & 下采样
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.downsample = nn.MaxPool2d(2)

        # 融合权重
        self.w_p4_td = FastWeightedFusion(2)
        self.w_p3_td = FastWeightedFusion(2)

        self.w_p4_out = FastWeightedFusion(3)
        self.w_p5_out = FastWeightedFusion(2)

        # 卷积（全部用可分离卷积）
        self.conv_p3 = SeparableConv(c)
        self.conv_p4 = SeparableConv(c)
        self.conv_p5 = SeparableConv(c)

    def forward(self, P3, P4, P5):
        # -----------------------
        # Top-down pathway
        # -----------------------
        P5_up = self.upsample(P5)

        P4_td = self.w_p4_td([P4, P5_up])
        P4_td = self.conv_p4(P4_td)

        P4_up = self.upsample(P4_td)

        P3_td = self.w_p3_td([P3, P4_up])
        P3_td = self.conv_p3(P3_td)

        # -----------------------
        # Bottom-up pathway
        # -----------------------
        P3_down = self.downsample(P3_td)

        P4_out = self.w_p4_out([P4, P4_td, P3_down])
        P4_out = self.conv_p4(P4_out)

        P4_down = self.downsample(P4_out)

        P5_out = self.w_p5_out([P5, P4_down])
        P5_out = self.conv_p5(P5_out)

        return P3_td, P4_out, P5_out


class BiFPN(nn.Module):
    def __init__(self, c, n_layers=3):
        super().__init__()
        self.layers = nn.ModuleList(
            [BiFPN_Layer(c) for _ in range(n_layers)]
        )

    def forward(self, P3, P4, P5):
        for layer in self.layers:
            P3, P4, P5 = layer(P3, P4, P5)
        return P3, P4, P5