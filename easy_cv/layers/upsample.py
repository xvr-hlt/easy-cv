from torch import nn


def get_upsample_layer(inplanes, outplanes):
    return nn.Sequential(
        nn.UpsamplingNearest2d(scale_factor=2),
        nn.Conv2d(inplanes, outplanes, kernel_size=3, padding=1),
        nn.BatchNorm2d(outplanes),
        nn.ReLU(inplace=True)
    )
