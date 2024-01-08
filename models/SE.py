from torch import nn


class SEBlock(nn.Module):
    """ Squeeze-and-Excitation Block """
    def __init__(self, arg):
        super(SEBlock, self).__init__()
        self.deep_dim = arg.deep_dim
        self.conv1 = nn.Conv2d(self.deep_dim, self.deep_dim, 3, 1, 1)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.linear = nn.Linear(self.deep_dim, self.deep_dim)
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

        self.init_params()

    def init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight.data)
                if m.bias is not None:
                    nn.init.constant_(m.bias.data, 0.0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight.data)
                if isinstance(m, nn.Linear) and m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        u = self.conv1(x)
        c = self.avg_pool(u)
        c = c.view(c.size(0), -1)
        c = self.relu(c)
        c = self.linear(c)
        c = self.sigmoid(c)
        c = c.view(c.size(0), c.size(1), 1, 1)
        s = u * c
        return s

