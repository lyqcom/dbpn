import math
import mindspore
import mindspore.nn as nn
import numpy as np
import mindspore.ops as ops
from base_network import *


class Discriminator(nn.Cell):
    def __init__(self, num_channels, base_filter, image_size):
        super(Discriminator, self).__init__()
        self.image_size = image_size

        self.input_conv = ConvBlock(num_channels, base_filter, 3, 1, 1, activation='lrelu', norm=None)

        self.conv_blocks = nn.SequentialCell(
            ConvBlock(base_filter, base_filter, 3, 2, 1, activation='lrelu', norm='batch'),
            ConvBlock(base_filter, base_filter * 2, 3, 1, 1, activation='lrelu', norm='batch'),
            ConvBlock(base_filter * 2, base_filter * 2, 3, 2, 1, activation='lrelu', norm='batch'),
            ConvBlock(base_filter * 2, base_filter * 4, 3, 1, 1, activation='lrelu', norm='batch'),
            ConvBlock(base_filter * 4, base_filter * 4, 3, 2, 1, activation='lrelu', norm='batch'),
            ConvBlock(base_filter * 4, base_filter * 8, 3, 1, 1, activation='lrelu', norm='batch'),
            ConvBlock(base_filter * 8, base_filter * 8, 3, 2, 1, activation='lrelu', norm='batch'),
        )

        self.dense_layers = nn.SequentialCell(
            DenseBlock(base_filter * 8 * image_size // 16 * image_size // 16, base_filter * 16, activation='lrelu',
                       norm=None),
            DenseBlock(base_filter * 16, 1, activation='sigmoid', norm=None)
        )

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
        out = self.input_conv(x)
        out = self.conv_blocks(out)
        reshape = ops.Reshape()
        out = reshape(out, (out.shape[0], -1))
        out = self.dense_layers(out)
        return out


class FeatureExtractor(nn.Cell):
    def __init__(self, netVGG, feature_layer=[9, 18, 27, 36]):
        super(FeatureExtractor, self).__init__()
        # self.features = nn.SequentialCell(*list(netVGG.features.children()))
        # self.features = nn.SequentialCell(*list(netVGG.layers[0:44]))
        self.features = list(netVGG.layers)
        # self.features = nn.SequentialCell(*list(netVGG.layers[0:36]))
        self.feature_layer = feature_layer

    def construct(self, x):
        results = []
        for ii, model in enumerate(self.features):
            if ii in self.feature_layer:
                x = model(x)
                results.append(x)
        return results


class FeatureExtractorResnet(nn.Cell):
    def __init__(self, resnet):
        super(FeatureExtractorResnet, self).__init__()
        self.features = nn.SequentialCell(*list(resnet.children())[:-1])

    def construct(self, x):
        results = []
        results.append(self.features(x))
        return results