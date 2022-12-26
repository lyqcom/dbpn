import math
import mindspore
import mindspore.nn as nn
import numpy as np
from base_network import *

class Net(nn.Cell):
    def __init__(self, num_channels, base_filter, feat, num_stages, scale_factor):
        super(Net, self).__init__()

        if scale_factor == 2:
            kernel = 6
            stride = 2
            padding = 2
        elif scale_factor == 4:
            kernel = 8
            stride = 4
            padding = 2
        elif scale_factor == 8:
            kernel = 12
            stride = 8
            padding = 2

        # Initial Feature Extraction
        self.feat0 = ConvBlock(num_channels, feat, 3, 1, 1, activation='relu', norm=None)
        self.feat1 = ConvBlock(feat, base_filter, 1, 1, 0, activation='relu', norm=None)
        # Back-projection stages
        self.up1 = UpBlock(base_filter, kernel, stride, padding)
        self.down1 = DownBlock(base_filter, kernel, stride, padding)
        self.up2 = UpBlock(base_filter, kernel, stride, padding)
        # Reconstruction
        self.output_conv = ConvBlock(num_stages * base_filter, num_channels, 3, 1, 1, activation=None, norm=None)

        for m in self.cells_and_names():
            classname = m.__class__.__name__
            # print("minspore不支持torch.nn.init.kaiming_normal_此写法")
            if classname.find('Conv2d') != -1:
                mindspore.common.initializer.HeNormal(m.weight)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif classname.find('Conv2dTranspose') != -1:
                mindspore.common.initializer.HeNormal(m.weight)
                if m.bias is not None:
                    m.bias.data.zero_()

    def construct(self, x):
        x = self.feat0(x)
        x = self.feat1(x)

        h1 = self.up1(x)
        h2 = self.up2(self.down1(h1))
        op = mindspore.ops.Concat(1)
        x = self.output_conv(op(h2, h1))

        return x
