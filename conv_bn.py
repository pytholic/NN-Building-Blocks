def conv_bn(
    in_channels,
    out_channels,
    stride,
    activation=nn.ReLU,
):
    return nn.Sequential(
        nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        ),
        nn.BatchNorm2d(out_channels),
        activation(inplace=True),
    )
