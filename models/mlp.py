import math
import torch
import numpy as np
import torch.nn as nn
from torch.nn.parameter import Parameter
import torch.nn.functional as F
from collections import OrderedDict


class Sequential(nn.Sequential):
    def __init__(self, *args):
        super(Sequential, self).__init__(*args)
    def forward(self, x, *args):
        out = x
        for key, m in self.named_children():
            if '_relu' in key or '_tanh' in key:
                out = m(out)
            else:
                out = m(out,*args)
        return out

class MLPRegressor(nn.Module):
    def __init__(self, num_neuron):
        super().__init__()
        kwargs = dict()
        Conv2d = nn.Conv2d
        BatchNorm2d = nn.BatchNorm2d
        Linear = nn.Linear
        
        self.regressor = Sequential()
        for idx in range(1,len(num_neuron)-1):
            self.regressor.add_module('fc{}'.format(idx-1), Linear(num_neuron[idx-1], num_neuron[idx], **kwargs))
            self.regressor.add_module('fc{}_tanh'.format(idx), nn.Tanh())
            # self.regressor.add_module('fc{}_relu'.format(idx-1), nn.ReLU(inplace=True))
            # self.regressor.add_module('fc{}_dropout'.format(idx), nn.Dropout())
        # self.regressor.add_module('fc{}'.format(len(num_neuron)), Linear(num_neuron[-1],1))
        if len(num_neuron)>2:
            self.regressor.add_module('fc{}'.format(len(num_neuron)-2), Linear(num_neuron[-2], num_neuron[-1], **kwargs))

        # # Initialization
        for name, param in self.named_parameters():
            if 'fc' in name:
                if 'weight' in name:
                    nn.init.kaiming_normal_(param)
                elif 'bias' in name:
                    nn.init.uniform_(param, -0.1, 0.1)
        
    
    def forward(self, x, *args):
        x = x.view(x.size(0), -1)
        return self.regressor(x,*args)


class MLPClassifier(nn.Module):
    def __init__(self, num_input, num_neuron, num_classes):
        super().__init__()
        kwargs = dict()
        Conv2d = nn.Conv2d
        BatchNorm2d = nn.BatchNorm2d
        Linear = nn.Linear
        
        self.classifier = Sequential(OrderedDict([
            ('fc0', Linear(num_input, num_neuron[0], **kwargs)),
            # ('fc0_tanh', nn.Tanh())
            ('fc0_relu', nn.ReLU(inplace=True))
        ]))
        for idx in range(1,len(num_neuron)):
            self.classifier.add_module('fc{}'.format(idx), Linear(num_neuron[idx-1], num_neuron[idx], **kwargs))
            # self.classifier.add_module('fc{}_tanh'.format(idx), nn.Tanh())
            self.classifier.add_module('fc{}_relu'.format(idx), nn.ReLU(inplace=True))
            # self.classifier.add_module('fc{}_dropout'.format(idx), nn.Dropout())
        self.classifier.add_module('fc_out', Linear(num_neuron[-1], num_classes, **kwargs))
        
        for name, param in self.named_parameters():
            if 'fc' in name:
                if 'weight' in name:
                    nn.init.kaiming_normal_(param)
                elif 'bias' in name:
                    # nn.init.kaiming_normal_(param)
                    param.data.fill_(0)
            elif 'norm' in name and 'weight' in name:
                param.data.fill_(1)
            elif 'norm' in name and 'bias' in name:
                param.data.fill_(0)
            elif 'classifier' in name and 'bias' in name:
                param.data.fill_(0)

    
    def forward(self, x, *args):
        x = x.view(x.size(0), -1)
        return self.classifier(x,*args)

def mlp16(num_inputs=784, num_classes=10):
    return MLPClassifier(num_input=num_inputs, num_neuron=[16],num_classes=num_classes)

def mlp1000(num_inputs=784, num_classes=10):
    return MLPClassifier(num_input=num_inputs, num_neuron=[1000],num_classes=num_classes)
    

class LeNet5(nn.Module):
    def __init__(self, num_channels=1, image_size=28, num_classes=10):
        super().__init__()
        self.features = Sequential()
        init_padding = 2 if image_size==28 else 0
        self.features.add_module('conv1',  nn.Conv2d(num_channels, 6, kernel_size=5, stride=1, padding=init_padding))
        self.features.add_module('conv1_relu', nn.ReLU(inplace=True))
        self.features.add_module('conv1_pool', nn.MaxPool2d(kernel_size=2, stride=2, padding=0))
        self.features.add_module('conv2',  nn.Conv2d(6, 16, kernel_size=5, stride=1, padding=0))
        self.features.add_module('conv2_relu', nn.ReLU(inplace=True))
        self.features.add_module('conv2_pool', nn.MaxPool2d(kernel_size=2, stride=2, padding=0))

        self.classifier = Sequential()
        self.classifier.add_module('fc1', nn.Linear(16*5*5, 120))
        self.classifier.add_module('fc1_relu', nn.ReLU(inplace=True))
        self.classifier.add_module('fc2', nn.Linear(120, 84))
        self.classifier.add_module('fc2_relu', nn.ReLU(inplace=True))
        
        # last fc layer
        self.classifier.add_module('fc3', nn.Linear(84,num_classes))

    def forward(self, x):
        features = self.features(x)
        # features = features.view(features.size(0), -1)
        features = features.view(features.size(0), -1)
        out = self.classifier(features)

        return F.log_softmax(out, dim=1)
        # return out


class AlexNet(nn.Module):

    def __init__(self, num_channels=3, num_classes=10):
        super(AlexNet, self).__init__()
        self.features = Sequential(
            nn.Conv2d(num_channels, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.classifier = Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), 256 * 6 * 6)
        x = self.classifier(x)
        return x