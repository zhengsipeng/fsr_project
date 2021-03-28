import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
from ..utils import Exchange, BatchNorm2dParallel, ModuleParallel

__all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
           'resnet152']

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}


def conv3x3(in_planes, out_planes, stride=1, bias=False):
    "3x3 convolution with padding"
    return ModuleParallel(nn.Conv2d(in_planes, out_planes, kernel_size=3,
                                    stride=stride, padding=1, bias=bias))


def conv1x1(in_planes, out_planes, stride=1, bias=False):
    "1x1 convolution"
    return ModuleParallel(nn.Conv2d(in_planes, out_planes, kernel_size=1,
                                    stride=stride, padding=0, bias=bias))


class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, inplanes, planes, num_parallel, bn_threshold, fusion=None, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        #self.bn1 = BatchNorm2dParallel(planes, num_parallel)
        self.bn1 = ModuleParallel(nn.BatchNorm2d(planes))
        self.relu = ModuleParallel(nn.ReLU(inplace=True))
        self.conv2 = conv3x3(planes, planes)        
        self.num_parallel = num_parallel
        self.downsample = downsample
        self.stride = stride
        self.fusion = fusion

        if self.fusion == 'cen':
            self.bn2 = BatchNorm2dParallel(planes, num_parallel)
            self.exchange = Exchange()
            self.bn_threshold = bn_threshold
            self.bn2_list = []
            for module in self.bn2.modules():
                if isinstance(module, nn.BatchNorm2d):
                    self.bn2_list.append(module)
        elif self.fusion is None:
            self.bn2 = ModuleParallel(nn.BatchNorm2d(planes))

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
   
        if self.fusion == 'cen':
            out = self.exchange(out, self.bn2_list, self.bn_threshold)

        if self.downsample is not None:
            identity = self.downsample(x)

        out = [out[l] + identity[l] for l in range(self.num_parallel)]
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4
    def __init__(self, inplanes, planes, num_parallel, bn_threshold, fusion=None, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = conv1x1(inplanes, planes)
        self.bn1 = BatchNorm2dParallel(planes, num_parallel)
        self.conv2 = conv3x3(planes, planes, stride=stride)
        self.bn2 = BatchNorm2dParallel(planes, num_parallel)
        self.conv3 = conv1x1(planes, planes * 4)
        self.bn3 = BatchNorm2dParallel(planes * 4, num_parallel)
        self.relu = ModuleParallel(nn.ReLU(inplace=True))
        self.num_parallel = num_parallel
        self.downsample = downsample
        self.stride = stride
        self.fusion = fusion

        if self.fusion == 'cen':  # channel exchange        
            self.exchange = Exchange()
            self.bn_threshold = bn_threshold
            self.bn2_list = []
            for module in self.bn2.modules():
                if isinstance(module, nn.BatchNorm2d):
                    self.bn2_list.append(module)

    def forward(self, x):
        identity = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        if self.fusion == 'cen':
            out = self.exchange(out, self.bn2_list, self.bn_threshold)

        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out = [out[l] + identity[l] for l in range(self.num_parallel)]
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    def __init__(self, block, layers, num_parallel, bn_threshold=2e-2, fusion=None, zero_init_residual=False):
        super(ResNet, self).__init__()
        self.num_classes = 1000
        self.num_parallel = num_parallel
        self.inplanes = 64
        self.conv1 = ModuleParallel(nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,  bias=False))
        #self.bn1 = BatchNorm2dParallel(64, num_parallel)
        self.bn1 = ModuleParallel(nn.BatchNorm2d(64))
        self.relu = ModuleParallel(nn.ReLU(inplace=True))
        self.fusion = fusion
        self.maxpool = ModuleParallel(nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
        self.layer1 = self._make_layer(block, 64, layers[0], bn_threshold)
        self.layer2 = self._make_layer(block, 128, layers[1], bn_threshold, stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], bn_threshold, stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], bn_threshold, stride=2)
        self.avgpool = ModuleParallel(nn.AdaptiveAvgPool2d((1, 1)))
        self.fc = nn.Linear(512 * block.expansion, self.num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, bn_threshold, stride=1):
        # delete the last conv1x1 of ResNet
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion: #and not del_last:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride=stride),
                #BatchNorm2dParallel(planes * block.expansion, self.num_parallel)
                ModuleParallel(nn.BatchNorm2d(planes * block.expansion))
            )

        layers = []
        layers.append(block(self.inplanes, planes, self.num_parallel, bn_threshold, self.fusion, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, self.num_parallel, bn_threshold, self.fusion))

        return nn.Sequential(*layers)

    def forward(self, x):
        C_1 = self.conv1(x)
        C_1 = self.bn1(C_1)
        C_1 = self.relu(C_1)
        C_1 = self.maxpool(C_1)
        C_2 = self.layer1(C_1)
        C_3 = self.layer2(C_2)
        C_4 = self.layer3(C_3)
        C_5 = self.layer4(C_4)
        x = self.avgpool(C_5)
        return x


def maybe_download(model_name, model_url, model_dir=None, map_location=None):
    import os, sys
    from six.moves import urllib
    if model_dir is None:
        torch_home = os.path.expanduser(os.getenv('TORCH_HOME', '~/.torch'))
        model_dir = os.getenv('TORCH_MODEL_ZOO', os.path.join(torch_home, 'models'))
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    filename = '{}.pth.tar'.format(model_name)
    cached_file = os.path.join(model_dir, filename)
    if not os.path.exists(cached_file):
        url = model_url
        sys.stderr.write('Downloading: "{}" to {}\n'.format(url, cached_file))
        urllib.request.urlretrieve(url, cached_file)
    return torch.load(cached_file, map_location=map_location)


def expand_model_dict(model_dict, state_dict, num_parallel):
    model_dict_keys = model_dict.keys()
    state_dict_keys = state_dict.keys()
    for model_dict_key in model_dict_keys:
        model_dict_key_re = model_dict_key.replace("module.", '')
        if model_dict_key_re in state_dict_keys:
            model_dict[model_dict_key] = state_dict[model_dict_key_re]
        for i in range(num_parallel):
            bn = '.bn_%d' % i 
            replace = True if bn in model_dict_key_re else False
            model_dict_key_re = model_dict_key_re.replace(bn, '')
            if replace and model_dict_key_re in state_dict_keys:
                model_dict[model_dict_key] = state_dict[model_dict_key_re]
    return model_dict


def resnet18(pretrained=False, parallel=1, hr_pretrained=False, **kwargs):
    """Constructs a ResNet-18 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    if pretrained:
        # strict = False as we don't need fc layer params.
        if hr_pretrained:
            model.load_state_dict(torch.load("backbone/weights/resnet18_hr_10.pth"), strict=False)
        else:
            print('Loading the high resolution pretrained model ...')
            state_dict = maybe_download('resnet18', model_urls['resnet18'])
            model_dict = expand_model_dict(model.state_dict(), state_dict, parallel)
            model.load_state_dict(model_dict, strict=True)
            #model.load_state_dict(model_zoo.load_url(model_urls['resnet18']), strict=False)
    #for param in model.parameters():
    #    print(param)
    #    assert 1==0
    return model


def resnet34(pretrained=False, parallel=1, **kwargs):
    """Constructs a ResNet-34 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)
    if pretrained:
        state_dict = maybe_download('resnet34', model_urls['resnet34'])
        model_dict = expand_model_dict(model.state_dict(), state_dict, parallel)
        model.load_state_dict(model_dict, strict=True)
        #model.load_state_dict(model_zoo.load_url(model_urls['resnet34']), strict=False)
    return model


def resnet50(pretrained=False, parallel=1, **kwargs):
    """Constructs a ResNet-50 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    if pretrained:
        state_dict = maybe_download('resnet50', model_urls['resnet50'])
        model_dict = expand_model_dict(model.state_dict(), state_dict, parallel)
        model.load_state_dict(model_dict, strict=True)
        model.load_state_dict(model_zoo.load_url(model_urls['resnet50']), strict=False)
    return model


def resnet101(pretrained=False, parallel=1, **kwargs):
    """Constructs a ResNet-101 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)
    if pretrained:
        state_dict = maybe_download('resnet101', model_urls['resnet101'])
        model_dict = expand_model_dict(model.state_dict(), state_dict, parallel)
        model.load_state_dict(model_dict, strict=True)
        model.load_state_dict(model_zoo.load_url(model_urls['resnet101']), strict=False)
    return model


def resnet152(pretrained=False, parallel=1, **kwargs):
    """Constructs a ResNet-152 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 8, 36, 3], **kwargs)
    if pretrained:
        state_dict = maybe_download('resnet152', model_urls['resnet152'])
        model_dict = expand_model_dict(model.state_dict(), state_dict, parallel)
        model.load_state_dict(model_dict, strict=True)
        model.load_state_dict(model_zoo.load_url(model_urls['resnet152']))
    return model


if __name__=='__main__':
    #model = torchvision.models.resnet50()
    print("found ", torch.cuda.device_count(), " GPU(s)")
    device = torch.device("cuda")
    model = resnet101(detection=True).to(device)
    print(model)

    input = torch.randn(1, 3, 512, 512).to(device)
    output = model(input)