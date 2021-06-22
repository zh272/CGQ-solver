"""
    PreResNet model definition
    ported from https://github.com/bearpaw/pytorch-classification/blob/master/models/cifar/preresnet.py
"""

import math
import torch.nn as nn

__all__ = ['resnet56', 'resnet110', 'resnet164']


class Sequential(nn.Sequential):
    def __init__(self, *args):
        super(Sequential, self).__init__(*args)
    def forward(self, x, *args):
        out = x
        for key, m in self.named_children():
            out = m(out,*args)
        return out

        
class BasicBlockCurve(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlockCurve, self).__init__()
        kwargs = dict()
        Conv2d = nn.Conv2d
        BatchNorm2d = nn.BatchNorm2d

        self.bn1 = BatchNorm2d(inplanes, **kwargs)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = Conv2d(
            inplanes, planes, kernel_size=3, stride=stride, 
            padding=1, bias=False, **kwargs
        )

        self.bn2 = BatchNorm2d(planes, **kwargs)
        self.conv2 = Conv2d(
            planes, planes, kernel_size=3, stride=stride, 
            padding=1, bias=False, **kwargs
        )
        self.downsample = downsample
        self.stride = stride

    def forward(self, x, *args):
        residual = x

        out = self.bn1(x, *args)
        out = self.relu(out)
        out = self.conv1(out, *args)

        out = self.bn2(out, *args)
        out = self.relu(out)
        out = self.conv2(out, *args)

        if self.downsample is not None:
            residual = self.downsample(x, *args)

        out += residual

        return out


class BottleneckCurve(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BottleneckCurve, self).__init__()
        kwargs = dict()
        Conv2d = nn.Conv2d
        BatchNorm2d = nn.BatchNorm2d

        self.bn1 = BatchNorm2d(inplanes, **kwargs)
        self.conv1 = Conv2d(inplanes, planes, kernel_size=1, bias=False,
                                   **kwargs)
        self.bn2 = BatchNorm2d(planes, **kwargs)
        self.conv2 = Conv2d(planes, planes, kernel_size=3, stride=stride,
                                   padding=1, bias=False, **kwargs)
        self.bn3 = BatchNorm2d(planes, **kwargs)
        self.conv3 = Conv2d(planes, planes * 4, kernel_size=1, bias=False,
                                   **kwargs)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x, *args):
        residual = x

        out = self.bn1(x, *args)
        out = self.relu(out)
        out = self.conv1(out, *args)

        out = self.bn2(out, *args)
        out = self.relu(out)
        out = self.conv2(out, *args)

        out = self.bn3(out, *args)
        out = self.relu(out)
        out = self.conv3(out, *args)

        if self.downsample is not None:
            residual = self.downsample(x, *args)

        out += residual

        return out



class ResNet(nn.Module):

    def __init__(self, num_classes, depth=110):
        super(ResNet, self).__init__()
        kwargs = dict()
        Conv2d = nn.Conv2d
        BatchNorm2d = nn.BatchNorm2d
        Linear = nn.Linear

        if depth >= 44:
            assert (depth - 2) % 9 == 0, 'depth should be 9n+2'
            n = (depth - 2) // 9
            block = BottleneckCurve
        else:
            assert (depth - 2) % 6 == 0, 'depth should be 6n+2'
            n = (depth - 2) // 6
            block = BasicBlockCurve

        self.inplanes = 16
        self.conv1 = Conv2d(3, 16, kernel_size=3, padding=1,
                                   bias=False, **kwargs)
        self.layer1 = self._make_layer(block, 16, n, _Conv2d=Conv2d, **kwargs)
        self.layer2 = self._make_layer(block, 32, n, stride=2, _Conv2d=Conv2d, **kwargs)
        self.layer3 = self._make_layer(block, 64, n, stride=2, _Conv2d=Conv2d, **kwargs)
        self.bn = BatchNorm2d(64 * block.expansion, **kwargs)
        self.relu = nn.ReLU(inplace=True)
        # self.avgpool = nn.AvgPool2d(8)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = Linear(64 * block.expansion, num_classes, **kwargs)

        for name, param in self.named_parameters():
            if 'conv' in name and 'weight' in name:
                nn.init.kaiming_normal_(param, mode='fan_out', nonlinearity='relu')
            elif 'norm' in name and 'weight' in name:
                param.data.fill_(1)
            elif 'norm' in name and 'bias' in name:
                param.data.fill_(0)
            elif 'fc' in name and 'bias' in name:
                param.data.fill_(0)

    def _make_layer(self, block, planes, blocks, stride=1, _Conv2d=nn.Conv2d, **kwargs):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = _Conv2d(self.inplanes, planes * block.expansion, kernel_size=1,
                                       stride=stride, bias=False, **kwargs)

        layers = list()
        layers.append(block(self.inplanes, planes, stride=stride,
                            downsample=downsample, **kwargs))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, **kwargs))

        return Sequential(*layers)

    def forward(self, x, *args):
        x = self.conv1(x, *args)

        for block in self.layer1:  # 32x32
            x = block(x, *args)
        for block in self.layer2:  # 16x16
            x = block(x, *args)
        for block in self.layer3:  # 8x8
            x = block(x, *args)
        x = self.bn(x, *args)
        x = self.relu(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x, *args)

        return x


def resnet56(num_classes=10):
    return ResNet(num_classes=num_classes, depth=56)

def resnet110(num_classes=10):
    return ResNet(num_classes=num_classes, depth=110)

def resnet164(num_classes=10):
    return ResNet(num_classes=num_classes, depth=164)
