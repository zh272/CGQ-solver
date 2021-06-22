"""
    WideResNet model definition
    ported from https://github.com/meliketoy/wide-resnet.pytorch/blob/master/networks/wide_resnet.py
"""

import torch.nn as nn
import torch.nn.functional as F

__all__ = ['wrn_2810']



class Sequential(nn.Sequential):
    def __init__(self, *args):
        super(Sequential, self).__init__(*args)
    def forward(self, x, *args):
        out = x
        for key, m in self.named_children():
            out = m(out,*args)
        return out


class WideBasicCurve(nn.Module):
    def __init__(self, in_planes, planes, dropout_rate, stride=1):
        super(WideBasicCurve, self).__init__()
        kwargs = dict()
        Conv2d = nn.Conv2d
        BatchNorm2d = nn.BatchNorm2d

        self.bn1 = BatchNorm2d(in_planes, **kwargs)
        self.conv1 = Conv2d(in_planes, planes, kernel_size=3, padding=1, bias=False, **kwargs)
        self.dropout = nn.Dropout(p=dropout_rate)
        self.bn2 = BatchNorm2d(planes, **kwargs)
        self.conv2 = Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1,bias=False, **kwargs)
        
        self.relu = nn.ReLU(inplace=True)
        if stride != 1 or in_planes != planes:
            self.shortcut = Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=False, **kwargs)
        else:
            self.shortcut = None

    def forward(self, x, *args):
        residual = x

        out = self.bn1(x, *args)
        out = self.relu(out)
        out = self.conv1(out, *args)

        out = self.dropout(out)

        out = self.bn2(out, *args)
        out = self.relu(out)
        out = self.conv2(out, *args)

        if self.shortcut is not None:
            residual = self.shortcut(x, *args)

        out += residual

        return out


class WideResNet(nn.Module):
    def __init__(self, num_classes, depth=28, widen_factor=10, dropout_rate=0.):
        super(WideResNet, self).__init__()
            
        self.in_planes = 16

        assert ((depth - 4) % 6 == 0), 'Wide-resnet depth should be 6n+4'
        n = (depth - 4) / 6
        k = widen_factor

        nstages = [16, 16 * k, 32 * k, 64 * k]

        kwargs = dict()
        Conv2d = nn.Conv2d
        BatchNorm2d = nn.BatchNorm2d
        Linear = nn.Linear

        self.conv1 = Conv2d(
            3, nstages[0], kernel_size=3, stride=1, padding=1, bias=True, **kwargs
        )
        self.layer1 = self._wide_layer(nstages[1], n, dropout_rate, stride=1)
        self.layer2 = self._wide_layer(nstages[2], n, dropout_rate, stride=2)
        self.layer3 = self._wide_layer(nstages[3], n, dropout_rate, stride=2)
        self.bn1 = BatchNorm2d(nstages[3], momentum=0.9, **kwargs)
        self.linear = Linear(nstages[3], num_classes, **kwargs)
        self.relu = nn.ReLU(inplace=True)

        for name, param in self.named_parameters():
            if 'conv' in name and 'weight' in name:
                nn.init.kaiming_normal_(param, mode='fan_out', nonlinearity='relu')
            elif 'norm' in name and 'weight' in name:
                param.data.fill_(1)
            elif 'norm' in name and 'bias' in name:
                param.data.fill_(0)
            elif 'linear' in name and 'bias' in name:
                param.data.fill_(0)

            

    def _wide_layer(self, planes, num_blocks, dropout_rate, stride):
        strides = [stride] + [1] * int(num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(
                WideBasicCurve(self.in_planes, planes, dropout_rate, stride=stride)
            )
            self.in_planes = planes
        
        return Sequential(*layers)

    def forward(self, x, *args):
        x = self.conv1(x, *args)

        # x = self.layer1(x, *args)
        # x = self.layer2(x, *args)
        # x = self.layer3(x, *args)
        for block in self.layer1:
            x = block(x, *args)
        for block in self.layer2:
            x = block(x, *args)
        for block in self.layer3:
            x = block(x, *args)

        x = self.bn1(x, *args)
        x = self.relu(x)
        
        x = F.avg_pool2d(x, 8)
        x = x.view(x.size(0), -1)
        x = self.linear(x, *args)

        return x


    
def wrn_2810(num_classes=10):
    return WideResNet(
        num_classes, depth=28, widen_factor=10, dropout_rate=0.
    )

