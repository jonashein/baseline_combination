from collections import OrderedDict

import torch.nn as nn
import torch.utils.model_zoo as model_zoo
from .domainnorm import DomainNorm

__all__ = ["ResNet", "resnet18", "resnet34", "resnet50", "resnet101", "resnet152"]

model_urls = {
    "resnet18": "https://download.pytorch.org/models/resnet18-5c106cde.pth",
    "resnet34": "https://download.pytorch.org/models/resnet34-333f7ec4.pth",
    "resnet50": "https://download.pytorch.org/models/resnet50-19c8e357.pth",
    "resnet101": "https://download.pytorch.org/models/resnet101-5d3b4d8f.pth",
    "resnet152": "https://download.pytorch.org/models/resnet152-b121ed2d.pth",
}


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, domain_norm=False):
        self.domain_norm = domain_norm
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        if self.domain_norm:
            self.dn1 = DomainNorm(planes)
        else:
            self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        if self.domain_norm:
            self.dn2 = DomainNorm(planes)
        else:
            self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        if self.downsample is not None and self.domain_norm:
            self.dsdn = DomainNorm(planes)
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        if self.domain_norm:
            out = self.dn1(out)
        else:
            out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        if self.domain_norm:
            out = self.dn2(out)
        else:
            out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)
            if self.domain_norm:
                residual = self.dsdn(residual)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    def __init__(
        self, block, layers, num_classes=1000, features=True, early_features=False, return_inter=False,
            return_cuda_inter=False, remove_avg_pool_layer=False, fully_conv=False, domain_norm=False,
    ):
        self.inplanes = 64
        self.early_features = early_features
        self.return_inter = (return_inter or return_cuda_inter)
        self.return_cuda_inter = return_cuda_inter
        self.features = features
        self.remove_avg_pool_layer = remove_avg_pool_layer
        self.fully_conv = fully_conv
        self.domain_norm = domain_norm
        super(ResNet, self).__init__()

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        if self.domain_norm:
            self.dn1 = DomainNorm(64)
        else:
            self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0], domain_norm=domain_norm)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, domain_norm=domain_norm)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, domain_norm=domain_norm)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, domain_norm=domain_norm)
        self.avgpool = nn.AvgPool2d(7)
        if self.fully_conv:
            self.avgpool = nn.AvgPool2d(7, padding=3, stride=1)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


    def _make_layer(self, block, planes, blocks, stride=1, domain_norm=False):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            if domain_norm:
                downsample = nn.Sequential(
                    nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                )
            else:
                downsample = nn.Sequential(
                    nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                    nn.BatchNorm2d(planes * block.expansion),
                )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, domain_norm=domain_norm))

        return nn.Sequential(*layers)


    def _return_inter(self, ret, x, name):
        if self.return_inter:
            if self.return_cuda_inter:
                ret[name] = x
            else:
                ret[name] = x.cpu()


    def forward(self, x):
        if self.return_inter:
            intermediates = OrderedDict()

        x = self.conv1(x)
        if self.domain_norm:
            x = self.dn1(x)
        else:
            x = self.bn1(x)
        x = self.relu(x)
        self._return_inter(intermediates, x, "res_conv1_relu")
        x = self.maxpool(x)
        x = self.layer1(x)
        self._return_inter(intermediates, x, "res_layer1")
        x = self.layer2(x)
        self._return_inter(intermediates, x, "res_layer2")
        x = self.layer3(x)
        self._return_inter(intermediates, x, "res_layer3")
        x = self.layer4(x)
        self._return_inter(intermediates, x, "res_layer4")
        if self.early_features:
            return x

        if not self.remove_avg_pool_layer:
            x = x.mean(3).mean(2)
        if not self.fully_conv:
            x = x.view(x.size(0), -1)

        if self.return_inter: # true
            return x, intermediates
        if self.features:
            return x, {}
        x = self.fc(x)

        return x


def resnet18(pretrained=False, **kwargs):
    """Constructs a ResNet-18 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls["resnet18"]))
    return model


def resnet34(pretrained=False, **kwargs):
    """Constructs a ResNet-34 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls["resnet34"]))
    return model


def resnet50(pretrained=False, **kwargs):
    """Constructs a ResNet-50 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls["resnet50"]))
    return model


def resnet101(pretrained=False, **kwargs):
    """Constructs a ResNet-101 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls["resnet101"]))
    return model


def resnet152(pretrained=False, **kwargs):
    """Constructs a ResNet-152 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 8, 36, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls["resnet152"]))
    return model
