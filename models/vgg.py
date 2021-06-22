"""
    VGG model definition
    ported from https://github.com/pytorch/vision/blob/master/torchvision/models/vgg.py
"""

import math
import torch.nn as nn

__all__ = ['VGG','vgg16']

config = {
    16: [[64, 64], [128, 128], [256, 256, 256], [512, 512, 512], [512, 512, 512]],
    19: [[64, 64], [128, 128], [256, 256, 256, 256], [512, 512, 512, 512], [512, 512, 512, 512]],
}


def make_layers(config, batch_norm=False, fix_points=None):
    layer_blocks = nn.ModuleList()
    activation_blocks = nn.ModuleList()
    poolings = nn.ModuleList()

    kwargs = dict()
    conv = nn.Conv2d
    bn = nn.BatchNorm2d
    in_channels = 3
    for sizes in config:
        layer_blocks.append(nn.ModuleList())
        activation_blocks.append(nn.ModuleList())
        for channels in sizes:
            layer_blocks[-1].append(conv(in_channels, channels, kernel_size=3, padding=1, **kwargs))
            if batch_norm:
                layer_blocks[-1].append(bn(channels, **kwargs))
            activation_blocks[-1].append(nn.ReLU(inplace=True))
            in_channels = channels
        poolings.append(nn.MaxPool2d(kernel_size=2, stride=2))
    return layer_blocks, activation_blocks, poolings


class VGGBase(nn.Module):
    def __init__(self, num_classes, depth=16, batch_norm=False):
        super(VGGBase, self).__init__()
        layer_blocks, activation_blocks, poolings = make_layers(config[depth], batch_norm)
        self.layer_blocks = layer_blocks
        self.activation_blocks = activation_blocks
        self.poolings = poolings

        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(512, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(512, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, num_classes),
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                m.bias.data.zero_()

    def forward(self, x):
        for layers, activations, pooling in zip(self.layer_blocks, self.activation_blocks,
                                                self.poolings):
            for layer, activation in zip(layers, activations):
                x = layer(x)
                x = activation(x)
            x = pooling(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


class VGGCurve(nn.Module):
    def __init__(self, num_classes, depth=16, batch_norm=False):
        super(VGGCurve, self).__init__()
        layer_blocks, activation_blocks, poolings = make_layers(config[depth],batch_norm)
        self.layer_blocks = layer_blocks
        self.activation_blocks = activation_blocks
        self.poolings = poolings

        self.dropout1 = nn.Dropout()
        self.fc1 = nn.Linear(512, 512)
        self.relu1 = nn.ReLU(inplace=True)
        self.dropout2 = nn.Dropout()
        self.fc2 = nn.Linear(512, 512)
        self.relu2 = nn.ReLU(inplace=True)
        self.fc3 = nn.Linear(512, num_classes)

        # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                for i in range(m.num_bends):
                    getattr(m, 'weight_%d' % i).data.normal_(0, math.sqrt(2. / n))
                    getattr(m, 'bias_%d' % i).data.zero_()

    def forward(self, x, coeffs_t):
        for layers, activations, pooling in zip(self.layer_blocks, self.activation_blocks,
                                                self.poolings):
            for layer, activation in zip(layers, activations):
                x = layer(x, coeffs_t)
                x = activation(x)
            x = pooling(x)
        x = x.view(x.size(0), -1)

        x = self.dropout1(x)
        x = self.fc1(x, coeffs_t)
        x = self.relu1(x)

        x = self.dropout2(x)
        x = self.fc2(x, coeffs_t)
        x = self.relu2(x)

        x = self.fc3(x, coeffs_t)

        return x

class VGG(nn.Module):
    def __init__(self, num_classes, depth=16, batch_norm=False):
        super(VGG, self).__init__()
        kwargs = dict()
        Conv2d = nn.Conv2d
        BatchNorm2d = nn.BatchNorm2d
        Linear = nn.Linear

        layer_blocks, activation_blocks, poolings = make_layers(config[depth], batch_norm)
        self.layer_blocks = layer_blocks
        self.activation_blocks = activation_blocks
        self.poolings = poolings

        self.dropout1 = nn.Dropout()
        self.fc1 = Linear(512, 512, **kwargs)
        self.relu1 = nn.ReLU(inplace=True)
        self.dropout2 = nn.Dropout()
        self.fc2 = Linear(512, 512, **kwargs)
        self.relu2 = nn.ReLU(inplace=True)
        self.fc3 = Linear(512, num_classes, **kwargs)


        # Initialize weights
        for m in self.modules():
            if isinstance(m, Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                if hasattr(m,'num_bends'):
                    for i in range(m.num_bends):
                        getattr(m, 'weight_%d' % i).data.normal_(0, math.sqrt(2. / n))
                        getattr(m, 'bias_%d' % i).data.zero_()
                else:
                    m.weight.data.normal_(0, math.sqrt(2. / n))
                    m.bias.data.zero_()

    def forward(self, x, *args):
        for layers, activations, pooling in zip(self.layer_blocks, self.activation_blocks,
                                                self.poolings):
            for layer, activation in zip(layers, activations):
                x = layer(x, *args)
                x = activation(x)
            x = pooling(x)
        x = x.view(x.size(0), -1)

        x = self.dropout1(x)
        x = self.fc1(x, *args)
        x = self.relu1(x)

        x = self.dropout2(x)
        x = self.fc2(x, *args)
        x = self.relu2(x)

        x = self.fc3(x, *args)

        return x

def vgg16(num_classes=10):
    return VGG(num_classes=num_classes, depth=16)