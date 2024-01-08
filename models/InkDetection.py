from torch import nn
import segmentation_models_pytorch as smp

from models.SE import SEBlock
from models.CNN3D import CNN3D
from models.AttnUNet import AttnUNet
from models.SegFormer import SegFormer_b0
from transformers import SegformerForSemanticSegmentation

class InkDetection(nn.Module):
    def __init__(self, arg):
        super(InkDetection, self).__init__()
        self.args = arg

        self.loss_weight = arg.loss_weight
        self.bce = smp.losses.SoftBCEWithLogitsLoss().to(arg.device)
        self.dice = smp.losses.DiceLoss(mode='binary').to(arg.device)

        if arg.use_se:
            self.SE = SEBlock(arg)
        if arg.use_cnn3d:
            self.CNN3D = CNN3D(arg)
        if arg.use_unet:
            self.UNet = AttnUNet(arg)
        if arg.my_segformer:
            # self.seg1 = MySegFormer(phi=arg.backbone, num_classes=1)
            self.seg1 = SegFormer_b0()
            self.upscaler1 = nn.ConvTranspose2d(1, 1, kernel_size=(4, 4), stride=2, padding=1)
            self.upscaler2 = nn.ConvTranspose2d(1, 1, kernel_size=(4, 4), stride=2, padding=1)
        if arg.library_segformer:
            try:
                self.seg2 = SegformerForSemanticSegmentation.from_pretrained(
                    "models/weights",
                    num_labels=1, ignore_mismatched_sizes=True, num_channels=16,)
            except OSError:
                self.seg2 = SegformerForSemanticSegmentation.from_pretrained(
                    "weights",
                    num_labels=1, ignore_mismatched_sizes=True, num_channels=16,)
            self.upscaler1 = nn.ConvTranspose2d(1, 1, kernel_size=(4, 4), stride=2, padding=1)
            self.upscaler2 = nn.ConvTranspose2d(1, 1, kernel_size=(4, 4), stride=2, padding=1)

        self.sigmoid = nn.Sigmoid()

        self.init_params()

    def init_params(self):
        for m in self.modules():
            if isinstance(m, nn.ConvTranspose2d):
                nn.init.normal_(m.weight.data)
                if m.bias is not None:
                    nn.init.constant_(m.bias.data, 0.0)

    def forward(self, x):
        if self.args.use_se:
            x = self.SE(x)
        if self.args.use_cnn3d:
            x = self.CNN3D(x)
        if self.args.use_unet:
            x = self.UNet(x)
        if self.args.my_segformer:
            x = self.seg1(x)
            x = self.upscaler1(x)
            x = self.upscaler2(x)
        if self.args.library_segformer:
            x = self.seg2(x).logits
            x = self.upscaler1(x)
            x = self.upscaler2(x)
        x = self.sigmoid(x)
        return x

    def criterion(self, pred, label):
        loss_bce = self.bce(pred, label)
        loss_dice = self.dice(pred, label)
        loss_total = self.loss_weight[0] * loss_bce + self.loss_weight[1] * loss_dice
        return loss_total


if __name__ == "__main__":
    import torch
    import argparse
    from torchinfo import summary
    from torchview import draw_graph

    parser = argparse.ArgumentParser(description='InkDetection')
    parser.add_argument('--deep_dim', type=int, default=16, help='deep dimension')
    parser.add_argument('--loss_weight', default=[0.3, 0.7], help='loss weight')
    parser.add_argument('--device', type=str, default=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    # SE Block
    parser.add_argument('--use_se', default=True, help='use se block or not')
    # CNN3D
    parser.add_argument('--use_cnn3d', default=False, help='use cnn3d block or not')
    # Attn UNet
    parser.add_argument('--use_unet', default=False, help='use unet block or not')
    parser.add_argument('--unet_channels', default=[16, 32, 64, 128, 256], help='unet channels')
    parser.add_argument('--use_attn', default=True, help='use attention block')
    # SegFormer
    parser.add_argument('--my_segformer', default=False, help='use segformer block or not')
    parser.add_argument('--library_segformer', default=True, help='use segformer block or not')
    parser.add_argument('--backbone', default='b0', help='backbone')
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = InkDetection(args).to(device)

    inputs = torch.randn(4, 16, 512, 512).to(device)
    out = model(inputs)
    print(out.shape)

    summary(model, input_size=inputs.shape, device=device, depth=10)

    model_graph = draw_graph(model, input_size=inputs.shape, device=device, save_graph=True, filename='InkDetection',
                             expand_nested=True, depth=10)