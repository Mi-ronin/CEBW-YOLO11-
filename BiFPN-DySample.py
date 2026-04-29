class BiFPN_DySample_Layer(nn.Module):
    def __init__(self, c):
        super().__init__()

        # DySample替代Upsample
        self.upsample = DySample(c)

        # 下采样
        self.downsample = nn.MaxPool2d(2)

        # 融合权重
        self.w_p4_td = FastWeightedFusion(2)
        self.w_p3_td = FastWeightedFusion(2)

        self.w_p4_out = FastWeightedFusion(3)
        self.w_p5_out = FastWeightedFusion(2)

        # 卷积
        self.conv_p3 = SeparableConv(c)
        self.conv_p4 = SeparableConv(c)
        self.conv_p5 = SeparableConv(c)

    def forward(self, P3, P4, P5):
        # -----------------------
        # Top-down
        # -----------------------
        P5_up = self.upsample(P5)

        P4_td = self.w_p4_td([P4, P5_up])
        P4_td = self.conv_p4(P4_td)

        P4_up = self.upsample(P4_td)

        P3_td = self.w_p3_td([P3, P4_up])
        P3_td = self.conv_p3(P3_td)

        # -----------------------
        # Bottom-up
        # -----------------------
        P3_down = self.downsample(P3_td)

        P4_out = self.w_p4_out([P4, P4_td, P3_down])
        P4_out = self.conv_p4(P4_out)

        P4_down = self.downsample(P4_out)

        P5_out = self.w_p5_out([P5, P4_down])
        P5_out = self.conv_p5(P5_out)

        return P3_td, P4_out, P5_out

    class BiFPN_DySample(nn.Module):
        def __init__(self, c, n_layers=3):
            super().__init__()
            self.layers = nn.ModuleList(
                [BiFPN_DySample_Layer(c) for _ in range(n_layers)]
            )

        def forward(self, P3, P4, P5):
            for layer in self.layers:
                P3, P4, P5 = layer(P3, P4, P5)
            return P3, P4, P5

