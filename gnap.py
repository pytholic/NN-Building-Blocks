class GNAP(Module):
    """Global Norm-Aware Pooling block"""

    def __init__(self, in_c):
        super(GNAP, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_c, affine=False)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.bn2 = nn.BatchNorm1d(in_c, affine=False)

    def forward(self, x):
        x = self.bn1(x)
        x_norm = torch.norm(x, 2, 1, True)
        x_norm_mean = torch.mean(x_norm)
        weight = x_norm_mean / x_norm
        x = x * weight
        x = self.pool(x)
        x = x.view(x.shape[0], -1)
        feature = self.bn2(x)
        return feature
