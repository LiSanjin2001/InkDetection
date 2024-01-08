import torch
from torch import nn


class DownBlock(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(DownBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, 3, 1, 1),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch_out, ch_out, 3, 1, 1),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )
        self.maxpool = nn.MaxPool2d(2, 2)

    def forward(self, x):
        skip_out = self.conv(x)
        next_out = self.maxpool(skip_out)
        return skip_out, next_out


class AttnBlock(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(AttnBlock, self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, 1, 1, 0, bias=True),
            nn.BatchNorm2d(ch_out)
        )
        self.W_x = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, 1, 1, 0, bias=True),
            nn.BatchNorm2d(ch_out)
        )
        self.psi = nn.Sequential(
            nn.Conv2d(ch_out, 1, 1, 1, 0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        out = x * psi
        return out


class UpBlock(nn.Module):
    def __init__(self, ch_in, ch_out, use_attn=True):
        super(UpBlock, self).__init__()

        self.attn_in = ch_in // 2
        self.attn_out = ch_out // 2
        self.use_attn = use_attn

        self.up = nn.Sequential(
            nn.ConvTranspose2d(ch_in, ch_out, 2, 2),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )
        self.attn = AttnBlock(self.attn_in, self.attn_out)
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, 3, 1, 1),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch_out, ch_out, 3, 1, 1),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self, x, skip_x):
        x = self.up(x)
        if self.use_attn:
            skip_x = self.attn(x, skip_x)
        x = torch.cat([skip_x, x], dim=1)
        x = self.conv(x)
        return x


class AttnUNet(nn.Module):
    def __init__(self, arg):
        super(AttnUNet, self).__init__()
        self.channels = arg.unet_channels
        self.use_attn = arg.use_attn
        self.deep_dim = arg.deep_dim

        self.down1 = DownBlock(self.deep_dim, self.channels[0])
        self.down2 = DownBlock(self.channels[0], self.channels[1])
        self.down3 = DownBlock(self.channels[1], self.channels[2])
        self.down4 = DownBlock(self.channels[2], self.channels[3])
        self.down5 = DownBlock(self.channels[3], self.channels[4])

        self.up4 = UpBlock(self.channels[4], self.channels[3], self.use_attn)
        self.up3 = UpBlock(self.channels[3], self.channels[2], self.use_attn)
        self.up2 = UpBlock(self.channels[2], self.channels[1], self.use_attn)
        self.up1 = UpBlock(self.channels[1], self.channels[0], self.use_attn)

        self.out = nn.Sequential(
            nn.Conv2d(16, 1, 3, 1, 1),
        )

        self.init_params()

    def init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight.data)
                if m.bias is not None:
                    nn.init.constant_(m.bias.data, 0.0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight.data, 1.0)
                nn.init.constant_(m.bias.data, 0.0)

    def forward(self, x):
        skip1, x = self.down1(x)
        skip2, x = self.down2(x)
        skip3, x = self.down3(x)
        skip4, x = self.down4(x)
        skip5, x = self.down5(x)

        x = self.up4(skip5, skip4)
        x = self.up3(x, skip3)
        x = self.up2(x, skip2)
        x = self.up1(x, skip1)

        x = self.out(x)
        return x


if __name__ == '__main__':
    import argparse
    from torchinfo import summary
    from torchview import draw_graph

    parser = argparse.ArgumentParser(description='UNet')
    parser.add_argument('--unet_channels', default=[16, 32, 64, 128, 256], help='unet channels')
    parser.add_argument('--use_attn', default=True, help='use attention block')
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AttnUNet(args).to(device)

    inputs = torch.randn(4, 3, 512, 512).to(device)
    outputs = model(inputs)
    print(outputs.shape)

    summary(model, (4, 3, 512, 512))

    mode_graph = draw_graph(model, inputs, device=device, save_graph=True, filename='AttnUNet', expand_nested=True)
