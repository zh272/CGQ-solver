import re
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
from collections import OrderedDict

__all__ = ['DenseNet', 'densenet100', 'densenet121', 'densenet169', 'densenet201', 'densenet161']


model_urls = {
    'densenet121': 'https://download.pytorch.org/models/densenet121-a639ec97.pth',
    'densenet169': 'https://download.pytorch.org/models/densenet169-b2777c0a.pth',
    'densenet201': 'https://download.pytorch.org/models/densenet201-c1103571.pth',
    'densenet161': 'https://download.pytorch.org/models/densenet161-8d451a50.pth',
}


class _DenseLayer(nn.Module):
    def __init__(self, num_input_features, growth_rate, bn_size, drop_rate):
        super(_DenseLayer, self).__init__()
        kwargs = dict()
        Conv2d = nn.Conv2d
        BatchNorm2d = nn.BatchNorm2d
        Linear = nn.Linear
        
        self.norm_1 = BatchNorm2d(num_input_features, **kwargs)
        self.relu_1 = nn.ReLU(inplace=True)
        self.conv_1 = Conv2d(num_input_features, bn_size *
                        growth_rate, kernel_size=1, stride=1, bias=False, **kwargs)
        self.norm_2 = BatchNorm2d(bn_size * growth_rate, **kwargs)
        self.relu_2 = nn.ReLU(inplace=True)
        self.conv_2 = Conv2d(bn_size * growth_rate, growth_rate,
                        kernel_size=3, stride=1, padding=1, bias=False, **kwargs)
        
        self.drop_rate = drop_rate

    def forward(self, x, *args):
        # new_features = super(_DenseLayer, self).forward(x)
        new_features = self.norm_1(x, *args)
        new_features = self.relu_1(new_features)
        new_features = self.conv_1(new_features, *args)
        new_features = self.norm_2(new_features, *args)
        new_features = self.relu_2(new_features)
        new_features = self.conv_2(new_features, *args)
        if self.drop_rate > 0:
            new_features = F.dropout(new_features, p=self.drop_rate, training=self.training)
        return torch.cat([x, new_features], 1)

class _DenseBlock(nn.Sequential):
    def __init__(self, num_layers, num_input_features, bn_size, growth_rate, drop_rate):
        super(_DenseBlock, self).__init__()
        kwargs = dict()
        for i in range(num_layers):
            layer = _DenseLayer(num_input_features + i * growth_rate, growth_rate, bn_size, drop_rate, **kwargs)
            self.add_module('denselayer%d' % (i + 1), layer)
    def forward(self, x, *args):
        out = x
        for key, m in self.named_children():
            out = m(out,*args)
        return out


# class _Transition(nn.Sequential):
class _Transition(nn.Module):
    def __init__(self, num_input_features, num_output_features):
        super(_Transition, self).__init__()
        kwargs = dict()
        Conv2d = nn.Conv2d
        BatchNorm2d = nn.BatchNorm2d
        Linear = nn.Linear
        
        # self.add_module('norm', BatchNorm2d(num_input_features, **kwargs))
        # self.add_module('relu', nn.ReLU(inplace=True))
        # self.add_module('conv', Conv2d(num_input_features, num_output_features,
        #                                   kernel_size=1, stride=1, bias=False, **kwargs))
        # self.add_module('pool', nn.AvgPool2d(kernel_size=2, stride=2))

        self.norm = BatchNorm2d(num_input_features, **kwargs)
        self.relu = nn.ReLU(inplace=True)
        self.conv = Conv2d(num_input_features, num_output_features,
                                          kernel_size=1, stride=1, bias=False, **kwargs)
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)
    def forward(self, x, *args):
        out = self.norm(x, *args)
        out = self.relu(out)
        out = self.conv(out, *args)
        out = self.pool(out)
        return out


class DenseNet(nn.Module):
    r"""Densenet-BC model class, based on
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`_
    Args:
        growth_rate (int) - how many filters to add each layer (`k` in paper)
        block_config (list of 4 ints) - how many layers in each pooling block
        num_init_features (int) - the number of filters to learn in the first convolution layer
        bn_size (int) - multiplicative factor for number of bottle neck layers
          (i.e. bn_size * k features in the bottleneck layer)
        drop_rate (float) - dropout rate after each dense layer
        num_classes (int) - number of classification classes
    """

    def __init__(self, growth_rate=32, block_config=(6, 12, 24, 16),
                 num_init_features=64, bn_size=4, drop_rate=0, num_classes=1000, 
                 small_inputs=True, init_method='kaiming'):

        super(DenseNet, self).__init__()
        kwargs = dict()
        Conv2d = nn.Conv2d
        BatchNorm2d = nn.BatchNorm2d
        Linear = nn.Linear

        # First convolution
        if small_inputs:
            self.features = nn.Sequential(OrderedDict([
                ('conv0', Conv2d(3, num_init_features, kernel_size=3, stride=1, padding=1, bias=False, **kwargs)),
            ]))
        else:
            self.features = nn.Sequential(OrderedDict([
                ('conv0', Conv2d(3, num_init_features, kernel_size=7, stride=2, padding=3, bias=False, **kwargs)),
                ('norm0', BatchNorm2d(num_init_features, **kwargs)),
                ('relu0', nn.ReLU(inplace=True)),
                ('pool0', nn.MaxPool2d(kernel_size=3, stride=2, padding=1)),
            ]))

        # Each denseblock
        num_features = num_init_features
        for i, num_layers in enumerate(block_config):
            block = _DenseBlock(num_layers=num_layers, num_input_features=num_features,
                                bn_size=bn_size, growth_rate=growth_rate, drop_rate=drop_rate, **kwargs)
            self.features.add_module('denseblock%d' % (i + 1), block)
            num_features = num_features + num_layers * growth_rate
            if i != len(block_config) - 1:
                trans = _Transition(num_input_features=num_features, num_output_features=num_features // 2, **kwargs)
                self.features.add_module('transition%d' % (i + 1), trans)
                num_features = num_features // 2

        # Final batch norm
        self.features.add_module('norm_final', BatchNorm2d(num_features, **kwargs))

        # Linear layer
        self.classifier = Linear(num_features, num_classes, **kwargs)

        # Official init from torch repo.
        # for m in self.modules():
        #     if isinstance(m, Conv2d):
        #         nn.init.kaiming_normal_(m.weight)
        #     elif isinstance(m, BatchNorm2d):
        #         nn.init.constant_(m.weight, 1)
        #         nn.init.constant_(m.bias, 0)
        #     elif isinstance(m, Linear):
        #         nn.init.constant_(m.bias, 0)
        for name, param in self.named_parameters():
            if 'conv' in name and 'weight' in name:
                if init_method=='xavier':
                    nn.init.xavier_normal_(param)
                else:
                    nn.init.kaiming_normal_(param)
            elif 'norm' in name and 'weight' in name:
                param.data.fill_(1)
            elif 'norm' in name and 'bias' in name:
                param.data.fill_(0)
            elif 'classifier' in name and 'bias' in name:
                param.data.fill_(0)

    def forward(self, x, *args):
        features = x
        # if coeffs_t is None:
        #     features = self.features(x)
        #     out = F.relu(features, inplace=True)
        #     out = F.adaptive_avg_pool2d(out, (1, 1)).view(features.size(0), -1)
        #     out = self.classifier(out)
        for key, m in self.features.named_children():
            if 'relu' in key or 'pool' in key:
                features = m(features)
            else:
                features = m(features, *args)
        out = F.relu(features, inplace=True)
        out = F.adaptive_avg_pool2d(out, (1, 1)).view(features.size(0), -1)
        out = self.classifier(out, *args)
        return out

def densenet100(num_classes=10, init_method='kaiming'):
    return DenseNet(growth_rate=12, block_config=(16, 16, 16),
                 num_init_features=24, bn_size=4, drop_rate=0,
                 num_classes=num_classes, small_inputs=True, init_method=init_method)

def densenet121(pretrained=False, **kwargs):
    r"""Densenet-121 model from
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = DenseNet(num_init_features=64, growth_rate=32, block_config=(6, 12, 24, 16),
                     **kwargs)
    if pretrained:
        # '.'s are no longer allowed in module names, but pervious _DenseLayer
        # has keys 'norm.1', 'relu.1', 'conv.1', 'norm.2', 'relu.2', 'conv.2'.
        # They are also in the checkpoints in model_urls. This pattern is used
        # to find such keys.
        pattern = re.compile(
            r'^(.*denselayer\d+\.(?:norm|relu|conv))\.((?:[12])\.(?:weight|bias|running_mean|running_var))$')
        state_dict = model_zoo.load_url(model_urls['densenet121'])
        for key in list(state_dict.keys()):
            res = pattern.match(key)
            if res:
                new_key = res.group(1) + res.group(2)
                state_dict[new_key] = state_dict[key]
                del state_dict[key]
        model.load_state_dict(state_dict)
    return model


def densenet169(pretrained=False, **kwargs):
    r"""Densenet-169 model from
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = DenseNet(num_init_features=64, growth_rate=32, block_config=(6, 12, 32, 32),
                     **kwargs)
    if pretrained:
        # '.'s are no longer allowed in module names, but pervious _DenseLayer
        # has keys 'norm.1', 'relu.1', 'conv.1', 'norm.2', 'relu.2', 'conv.2'.
        # They are also in the checkpoints in model_urls. This pattern is used
        # to find such keys.
        pattern = re.compile(
            r'^(.*denselayer\d+\.(?:norm|relu|conv))\.((?:[12])\.(?:weight|bias|running_mean|running_var))$')
        state_dict = model_zoo.load_url(model_urls['densenet169'])
        for key in list(state_dict.keys()):
            res = pattern.match(key)
            if res:
                new_key = res.group(1) + res.group(2)
                state_dict[new_key] = state_dict[key]
                del state_dict[key]
        model.load_state_dict(state_dict)
    return model


def densenet201(pretrained=False, **kwargs):
    r"""Densenet-201 model from
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = DenseNet(num_init_features=64, growth_rate=32, block_config=(6, 12, 48, 32),
                     **kwargs)
    if pretrained:
        # '.'s are no longer allowed in module names, but pervious _DenseLayer
        # has keys 'norm.1', 'relu.1', 'conv.1', 'norm.2', 'relu.2', 'conv.2'.
        # They are also in the checkpoints in model_urls. This pattern is used
        # to find such keys.
        pattern = re.compile(
            r'^(.*denselayer\d+\.(?:norm|relu|conv))\.((?:[12])\.(?:weight|bias|running_mean|running_var))$')
        state_dict = model_zoo.load_url(model_urls['densenet201'])
        for key in list(state_dict.keys()):
            res = pattern.match(key)
            if res:
                new_key = res.group(1) + res.group(2)
                state_dict[new_key] = state_dict[key]
                del state_dict[key]
        model.load_state_dict(state_dict)
    return model


def densenet161(pretrained=False, **kwargs):
    r"""Densenet-161 model from
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = DenseNet(num_init_features=96, growth_rate=48, block_config=(6, 12, 36, 24),
                     **kwargs)
    if pretrained:
        # '.'s are no longer allowed in module names, but pervious _DenseLayer
        # has keys 'norm.1', 'relu.1', 'conv.1', 'norm.2', 'relu.2', 'conv.2'.
        # They are also in the checkpoints in model_urls. This pattern is used
        # to find such keys.
        pattern = re.compile(
            r'^(.*denselayer\d+\.(?:norm|relu|conv))\.((?:[12])\.(?:weight|bias|running_mean|running_var))$')
        state_dict = model_zoo.load_url(model_urls['densenet161'])
        for key in list(state_dict.keys()):
            res = pattern.match(key)
            if res:
                new_key = res.group(1) + res.group(2)
                state_dict[new_key] = state_dict[key]
                del state_dict[key]
        model.load_state_dict(state_dict)
    return model