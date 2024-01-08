import math
import torch
from torch import nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_
from timm.models.layers import DropPath


class OverlapPatchEmbedding(nn.Module):
    def __init__(self, idx, **params):
        super(OverlapPatchEmbedding, self).__init__()
        patch_size = (params['patch_size'][idx], params['patch_size'][idx])
        if idx == 0:
            in_chans = params['in_chans']
        else:
            in_chans = params['embed_dim'][idx - 1]
        out_chans = params['embed_dim'][idx]
        stride = params['stride'][idx]
        self.proj = nn.Conv2d(in_chans, out_chans, kernel_size=patch_size, stride=stride,
                              padding=(patch_size[0] // 2, patch_size[1] // 2))
        self.norm = nn.LayerNorm(out_chans, eps=1e-6)

        self.init_params()

    def init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                fan_out //= m.groups
                m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        x = self.proj(x)
        _, _, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)
        return x, H, W


class EfficientAttention(nn.Module):
    def __init__(self, idx, **params):
        super(EfficientAttention, self).__init__()
        dim = params['embed_dim'][idx]
        self.sr_ratio = params['sr_ratio'][idx]
        self.num_heads = params['num_heads'][idx]
        self.scale = (dim // self.num_heads) ** -0.5

        self.sr = nn.Conv2d(dim, dim, kernel_size=self.sr_ratio, stride=self.sr_ratio)
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.value = nn.Linear(dim, dim, bias=True)
        self.key = nn.Linear(dim, dim, bias=True)
        self.query = nn.Linear(dim, dim, bias=True)
        self.drop = nn.Dropout(params['attn_drop_rate'])

    def forward(self, x, H, W):
        B, N, C = x.shape
        q = self.query(x)
        q = q.view(B, -1, self.num_heads, C // self.num_heads)
        q = q.permute(0, 2, 1, 3)
        if self.sr_ratio > 1:
            x = x.permute(0, 2, 1)
            x = x.reshape(B, C, H, W)
            x = self.sr(x)
            x = x.reshape(B, C, -1)
            x = x.permute(0, 2, 1)
            x = self.norm(x)
        k = self.key(x)
        k = k.view(B, -1, self.num_heads, C // self.num_heads)
        k = k.permute(0, 2, 1, 3)
        v = self.value(x)
        v = v.view(B, -1, self.num_heads, C // self.num_heads)
        v = v.permute(0, 2, 1, 3)
        k = k.transpose(2, 3)
        attn = torch.matmul(q, k)
        attn = attn * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.drop(attn)
        x = torch.matmul(attn, v)
        x = x.permute(0, 2, 1, 3)
        x = x.contiguous()
        x = x.view(B, -1, C)
        return x


class SegAttention(nn.Module):
    def __init__(self, idx, **params):
        super(SegAttention, self).__init__()
        self.effi = EfficientAttention(idx, **params)
        self.out = nn.Sequential(
            nn.Linear(params['embed_dim'][idx], params['embed_dim'][idx]),
            nn.Dropout(params['drop_rate']),
        )

        self.init_params()

    def init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                fan_out //= m.groups
                m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)
            elif isinstance(m, nn.Linear):
                trunc_normal_(m.weight.data, mean=0, std=0.02)
                if isinstance(m, nn.Linear) and m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x, H, W):
        x = self.effi(x, H, W)
        x = self.out(x)
        return x


class DWConv(nn.Module):
    def __init__(self, idx, **params):
        super(DWConv, self).__init__()
        chans = int(params['embed_dim'][idx] * params['mlp_ratio'])
        self.conv = nn.Conv2d(chans, chans, kernel_size=3, stride=1, padding=1, groups=chans)

    def forward(self, x, H, W):
        B, N, C = x.shape
        x = x.transpose(1, 2)
        x = x.view(B, C, H, W)
        x = self.conv(x)
        x = x.flatten(2).transpose(1, 2)
        return x


class MixFFN(nn.Module):
    def __init__(self, idx, **params):
        super(MixFFN, self).__init__()
        self.in_chans = params['embed_dim'][idx]
        self.hidden_chans = int(params['embed_dim'][idx] * params['mlp_ratio'])
        self.out_chans = params['embed_dim'][idx]

        self.fc1 = nn.Linear(self.in_chans, self.hidden_chans)
        self.dw = DWConv(idx, **params)
        self.gelu = nn.GELU()
        self.drop = nn.Dropout(params['drop_rate'])
        self.fc2 = nn.Linear(self.hidden_chans, self.out_chans)

        self.init_params()

    def init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                fan_out //= m.groups
                m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                trunc_normal_(m.weight.data, mean=0, std=0.02)
                if isinstance(m, nn.Linear) and m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x, H, W):
        x = self.fc1(x)
        x = self.dw(x, H, W)
        x = self.gelu(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class SegLayer(nn.Module):
    def __init__(self, idx, j, **params):
        super(SegLayer, self).__init__()
        dim = params['embed_dim'][idx]

        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.attn = SegAttention(idx, **params)
        self.drop_path = DropPath(params['drop_path_rate_list'][idx][j]) if params['drop_path_rate_list'][idx][
                                                                                j] > 0 else nn.Identity()
        self.ffn = MixFFN(idx, **params)

    def forward(self, in_data, H, W):
        x = self.norm(in_data)
        x = self.attn(x, H, W)
        x = self.drop_path(x)
        x1 = x + in_data
        x = self.norm(x1)
        x = self.ffn(x, H, W)
        x = self.drop_path(x)
        x = x + x1
        return x


class SegEncoder(nn.Module):
    def __init__(self, **params):
        super(SegEncoder, self).__init__()

        self.patch_embed1 = OverlapPatchEmbedding(idx=0, **params)
        self.seg_layer1_1 = SegLayer(idx=0, j=0, **params)
        self.seg_layer1_2 = SegLayer(idx=0, j=1, **params)
        self.layer_norm1 = nn.LayerNorm(params['embed_dim'][0], eps=1e-6)

        self.patch_embed2 = OverlapPatchEmbedding(idx=1, **params)
        self.seg_layer2_1 = SegLayer(idx=1, j=0, **params)
        self.seg_layer2_2 = SegLayer(idx=1, j=1, **params)
        self.layer_norm2 = nn.LayerNorm(params['embed_dim'][1], eps=1e-6)

        self.patch_embed3 = OverlapPatchEmbedding(idx=2, **params)
        self.seg_layer3_1 = SegLayer(idx=2, j=0, **params)
        self.seg_layer3_2 = SegLayer(idx=2, j=1, **params)
        self.layer_norm3 = nn.LayerNorm(params['embed_dim'][2], eps=1e-6)

        self.patch_embed4 = OverlapPatchEmbedding(idx=3, **params)
        self.seg_layer4_1 = SegLayer(idx=3, j=0, **params)
        self.seg_layer4_2 = SegLayer(idx=3, j=1, **params)
        self.layer_norm4 = nn.LayerNorm(params['embed_dim'][3], eps=1e-6)

        self.init_params()

    def init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                fan_out //= m.groups
                m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                trunc_normal_(m.weight.data, mean=0, std=0.02)
                if isinstance(m, nn.Linear) and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        outs = []

        x, H, W = self.patch_embed1(x)
        x = self.seg_layer1_1(x, H, W)
        x = self.seg_layer1_2(x, H, W)
        x = self.layer_norm1(x)
        x = x.reshape(x.shape[0], H, W, -1).permute(0, 3, 1, 2)
        x = x.contiguous()
        outs.append(x)

        x, H, W = self.patch_embed2(x)
        x = self.seg_layer2_1(x, H, W)
        x = self.seg_layer2_2(x, H, W)
        x = self.layer_norm2(x)
        x = x.reshape(x.shape[0], H, W, -1).permute(0, 3, 1, 2)
        x = x.contiguous()
        outs.append(x)

        x, H, W = self.patch_embed3(x)
        x = self.seg_layer3_1(x, H, W)
        x = self.seg_layer3_2(x, H, W)
        x = self.layer_norm3(x)
        x = x.reshape(x.shape[0], H, W, -1).permute(0, 3, 1, 2)
        x = x.contiguous()
        outs.append(x)

        x, H, W = self.patch_embed4(x)
        x = self.seg_layer4_1(x, H, W)
        x = self.seg_layer4_2(x, H, W)
        x = self.layer_norm4(x)
        x = x.reshape(x.shape[0], H, W, -1).permute(0, 3, 1, 2)
        x = x.contiguous()
        outs.append(x)

        return outs


class MLP(nn.Module):
    def __init__(self, idx, **params):
        super(MLP, self).__init__()
        self.fc = nn.Linear(params['embed_dim'][idx], params['embed_dim_out'])

    def forward(self, x):
        x = x.flatten(2).transpose(1, 2)
        x = self.fc(x)
        return x


class SegDecoder(nn.Module):
    def __init__(self, **params):
        super(SegDecoder, self).__init__()
        self.mlp1 = MLP(idx=0, **params)
        self.mlp2 = MLP(idx=1, **params)
        self.mlp3 = MLP(idx=2, **params)
        self.mlp4 = MLP(idx=3, **params)

        self.conv = nn.Conv2d(params['embed_dim_out'] * 4, params['embed_dim_out'], kernel_size=3, stride=1, padding=1)
        self.norm = nn.BatchNorm2d(params['embed_dim_out'])
        self.relu = nn.ReLU(inplace=True)
        self.drop = nn.Dropout(params['drop_rate'])
        self.conv_out = nn.Conv2d(params['embed_dim_out'], 1, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        c1, c2, c3, c4 = x
        B = c1.shape[0]

        x1 = self.mlp1(c1)
        x1 = x1.permute(0, 2, 1)
        x1 = x1.reshape(B, -1, c1.shape[2], c1.shape[3])
        x1 = F.interpolate(x1, size=x1.size()[2:], mode='bilinear', align_corners=True)

        x2 = self.mlp2(c2)
        x2 = x2.permute(0, 2, 1)
        x2 = x2.reshape(B, -1, c2.shape[2], c2.shape[3])
        x2 = F.interpolate(x2, size=x1.size()[2:], mode='bilinear', align_corners=True)

        x3 = self.mlp3(c3)
        x3 = x3.permute(0, 2, 1)
        x3 = x3.reshape(B, -1, c3.shape[2], c3.shape[3])
        x3 = F.interpolate(x3, size=x1.size()[2:], mode='bilinear', align_corners=True)

        x4 = self.mlp4(c4)
        x4 = x4.permute(0, 2, 1)
        x4 = x4.reshape(B, -1, c4.shape[2], c4.shape[3])
        x4 = F.interpolate(x4, size=x1.size()[2:], mode='bilinear', align_corners=True)

        x = torch.cat([x1, x2, x3, x4], dim=1)

        x = self.conv(x)
        x = self.norm(x)
        x = self.relu(x)
        x = self.drop(x)
        x = self.conv_out(x)

        return x


class SegFormer_b0(nn.Module):
    def __init__(self):
        super(SegFormer_b0, self).__init__()
        depths = [2, 2, 2, 2]
        drop_path_rate = 0.1
        drop_path_rate_list = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        dpr1 = drop_path_rate_list[:depths[0]]
        dpr2 = drop_path_rate_list[depths[0]:sum(depths[:2])]
        dpr3 = drop_path_rate_list[sum(depths[:2]):sum(depths[:3])]
        dpr4 = drop_path_rate_list[sum(depths[:3]):sum(depths[:4])]
        drop_path_rate_list = [dpr1, dpr2, dpr3, dpr4]

        params = {
            "patch_size": [7, 3, 3, 3],
            "stride": [4, 2, 2, 2],
            "in_chans": 16,
            "embed_dim": [32, 64, 160, 256],
            "sr_ratio": [8, 4, 2, 1],
            "num_heads": [1, 2, 5, 8],
            "attn_drop_rate": 0.0,
            "drop_rate": 0.0,
            "depths": depths,
            "drop_path_rate_list": drop_path_rate_list,
            "mlp_ratio": 4,
            "embed_dim_out": 256,
        }

        self.encoder = SegEncoder(**params)
        self.decoder = SegDecoder(**params)

        # self.convT1 = nn.ConvTranspose2d(1, 1, kernel_size=4, stride=2, padding=1)
        # self.convT2 = nn.ConvTranspose2d(1, 1, kernel_size=4, stride=2, padding=1)
        #
        # self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        # x = self.convT1(x)
        # x = self.convT2(x)
        # x = self.sigmoid(x)
        return x


if __name__ == "__main__":
    from torchinfo import summary
    from torchview import draw_graph

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SegFormer_b0().to(device)

    inputs = torch.randn(8, 16, 512, 512).to(device)
    out = model(inputs)
    print(out.shape)

    summary(model, input_size=inputs.shape, device=device, depth=10)

    model_graph = draw_graph(model, input_size=inputs.shape, device=device, save_graph=True, filename='Try',
                             expand_nested=True, depth=10)
