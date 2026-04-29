import torch
import torch.nn as nn
import torch.nn.functional as F


class DySample(nn.Module):

    def __init__(self, c, scale=2):
        super().__init__()
        self.scale = scale

        self.offset = nn.Conv2d(c, 2 * scale * scale, 1)
        self.mask = nn.Conv2d(c, scale * scale, 1)

        self.conv = nn.Conv2d(c, c, 1, bias=False)
        self.bn = nn.BatchNorm2d(c)
        self.act = nn.SiLU()

    def forward(self, x):
        B, C, H, W = x.shape

        # 上采样基准
        x_up = F.interpolate(x, scale_factor=self.scale, mode='bilinear', align_corners=False)

        # 偏移 & 权重
        offset = self.offset(x)
        mask = torch.sigmoid(self.mask(x))

        offset = F.pixel_shuffle(offset, self.scale)
        mask = F.pixel_shuffle(mask, self.scale)

        # 生成grid
        grid_y, grid_x = torch.meshgrid(
            torch.linspace(-1, 1, H * self.scale, device=x.device),
            torch.linspace(-1, 1, W * self.scale, device=x.device),
            indexing='ij'
        )
        base_grid = torch.stack((grid_x, grid_y), dim=-1)
        base_grid = base_grid.unsqueeze(0).repeat(B, 1, 1, 1)

        grid = base_grid + offset.permute(0, 2, 3, 1)

        # 动态采样
        sampled = F.grid_sample(x_up, grid, align_corners=False)

        # mask调制（关键点）
        out = sampled * mask

        return self.act(self.bn(self.conv(out)))