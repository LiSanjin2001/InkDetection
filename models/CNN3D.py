from torch import nn


class CNN3D(nn.Module):
    def __init__(self, arg):
        super(CNN3D,self).__init__()

        self.conv1 = nn.Conv3d(1, 2, 3, 1, 1,)
        self.conv2 = nn.Conv3d(2, 4, 3, 1, 1,)
        self.conv3 = nn.Conv3d(4, 8, 3, 1, 1,)
        self.conv4 = nn.Conv3d(8, 16, 3, 1, 1,)

        self.init_params()

    def init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight.data)
                if m.bias is not None:
                    nn.init.constant_(m.bias.data, 0.0)

    def forward(self,x):
        x = x.permute(0, 2, 3, 1)
        x = x.unsqueeze(1)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = x.max(axis=-1)[0]
        return x

if __name__ == '__main__':
    import torch
    import argparse
    from torchinfo import summary
    from torchview import draw_graph

    parser = argparse.ArgumentParser(description='InkDetection')
    parser.add_argument('--deep_dim', type=int, default=16, help='deep dimension')
    parser.add_argument('--dropout', type=float, default=0.2, help='dropout rate')
    parser.add_argument('--use_se', type=bool, default=True, help='use se block or not')
    parser.add_argument('--unet_channels', default=[16, 32, 64, 128, 256], help='unet channels')
    parser.add_argument('--use_attn', default=True, help='use attention block')
    parser.add_argument('--loss_weight', default=[0.3, 0.7], help='loss weight')
    parser.add_argument('--device', type=str, default=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    parser.add_argument('--backbone', default='b0', help='backbone')
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CNN3D(args).to(device)

    inputs = torch.randn(4, 16, 512, 512).to(device)
    out = model(inputs)
    print(out.shape)

    summary(model, input_size=inputs.shape, device=device, depth=10)

    model_graph = draw_graph(model, input_size=inputs.shape, device=device, save_graph=True, filename='InkDetection',
                             expand_nested=True, depth=10)