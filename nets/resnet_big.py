"""ResNet in PyTorch.
ImageNet-Style ResNet
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
Adapted from: https://github.com/bearpaw/pytorch-classification
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from .backbones.resnet import resnet18, resnet34, resnet50

from sklearn import metrics
from sklearn.svm import SVC, LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, is_last=False):
        super(BasicBlock, self).__init__()
        self.is_last = is_last
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        preact = out
        out = F.relu(out)
        if self.is_last:
            return out, preact
        else:
            return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1, is_last=False):
        super(Bottleneck, self).__init__()
        self.is_last = is_last
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion * planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion * planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        preact = out
        out = F.relu(out)
        if self.is_last:
            return out, preact
        else:
            return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, in_channel=3, zero_init_residual=False):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(in_channel, 64, kernel_size=3, stride=1, padding=1,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves
        # like an identity. This improves the model by 0.2~0.3% according to:
        # https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for i in range(num_blocks):
            stride = strides[i]
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x, layer=100):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        return out


def resnet18(**kwargs):
    return ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)


def resnet34(**kwargs):
    return ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)


def resnet50(**kwargs):
    return ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)


def resnet101(**kwargs):
    return ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)


model_dict = {
    'resnet18': [resnet18, 512],
    'resnet34': [resnet34, 512],
    'resnet50': [resnet50, 2048],
    'resnet101': [resnet101, 2048],
}


class LinearBatchNorm(nn.Module):
    """Implements BatchNorm1d by BatchNorm2d, for SyncBN purpose"""
    def __init__(self, dim, affine=True):
        super(LinearBatchNorm, self).__init__()
        self.dim = dim
        self.bn = nn.BatchNorm2d(dim, affine=affine)

    def forward(self, x):
        x = x.view(-1, self.dim, 1, 1)
        x = self.bn(x)
        x = x.view(-1, self.dim)
        return x


class SupConResNet(nn.Module):
    """backbone + projection head"""
    def __init__(self, args, head='mlp', feat_dim=512, num_gpus=4):
        super(SupConResNet, self).__init__()
        self.name = args.backbone
        model_fun, self.dim_in = model_dict[self.name]
        #self.encoder = model_fun()
        self.encoder = self.build_backbone()
        if head == 'linear':
            self.head = nn.Linear(self.dim_in, feat_dim)
        elif head == 'mlp':
            self.head = nn.Sequential(
                nn.Linear(self.dim_in, self.dim_in),
                nn.ReLU(inplace=True),
                nn.Linear(self.dim_in, feat_dim)
            )
        else:
            raise NotImplementedError(
                'head not supported: {}'.format(head))
        
        self.way = args.way
        self.shot = args.shot
        self.query = args.query
        self.classifier = args.classifier

        self.use_ce = args.use_ce
        self.num_classes = 64+12
        self.action_classifier = nn.Linear(self.dim_in, self.num_classes)

    def forward(self, x, aux=None, labels=None, is_pretrain=True):
        if is_pretrain:
            b2, t, c, h, w = x.shape
            x = x.reshape(b2*t, c, h, w)           
            bsz = int(b2/2)
            
            feat = self.encoder(x)
            #assert 1==0
            feat = feat.reshape(b2, t, -1).mean(1)  # b2, c
            logits = self.action_classifier(feat)

            feat = torch.flatten(feat, 1)
            feat = self.head(feat)
            #print(feat.shape)
            feat = F.normalize(feat, dim=1)
            f1, f2 = torch.split(feat, [bsz, bsz], dim=0)
            feat = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1)  # bsz, 2, c

            return logits, feat
        else:
            b, t, c, h, w = x.shape # way*(shot+query), num_frame, c, h, w
            x = x.view(b*t, c, h, w)
            x = self.encoder(x)
            #x = self.res_avgpool(x).squeeze()
            x = x.reshape(b, t, self.dim_in).mean(1)
            
            pivot = self.way * self.shot
            support_feats, query_feats = x[:pivot], x[pivot:] 
            support_labels = labels[:pivot]
            query_labels = labels[pivot:]

            query_pred = self.simple_classifier(self.classifier, support_feats, query_feats, support_labels, query_labels)

            return query_pred

    def build_backbone(self):
        if self.name == "resnet18":
            resnet = models.resnet18(pretrained=True)  
        elif self.name == "resnet34":
            resnet = models.resnet34(pretrained=True)
        elif self.name == "resnet50":
            resnet = models.resnet50(pretrained=True)

        resnet = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
            resnet.layer1,
            resnet.layer2,
            resnet.layer3,
            resnet.layer4,
            resnet.avgpool,  # we delete the pooling layer to output 7x7 feature map
        )

        return resnet

    def simple_classifier(self, classifier, support_feats, query_feats, support_labels, query_labels):
        support_feats = support_feats.cpu().numpy()
        query_feats = query_feats.cpu().numpy()
        support_labels = support_labels.cpu().numpy()
        query_labels = query_labels.cpu().numpy()
        
        if classifier == "LR":
            clf = LogisticRegression(penalty='l2', random_state=0, C=1.0, solver='lbfgs',
                                     max_iter=1000, multi_class='multinomial')
  
            clf.fit(support_feats, support_labels)
            query_pred = clf.predict(query_feats)
        elif classifier == 'SVM':
            clf = make_pipeline(StandardScaler(), SVC(gamma='auto', C=1, kernel='linear',
                                                    decision_function_shape='ovr'))
            clf.fit(support_feats, support_labels)
            query_pred = clf.predict(query_feats)
        elif classifier == 'NN':
            query_pred = NN(support_feats, support_labels, query_feats)
        elif classifier == 'Cosine':
            query_pred = Cosine(support_feats, support_labels, query_feats)
        elif classifier == 'Proto':
            query_pred = Proto(support_feats, support_labels, query_feats, self.way, self.shot)
        else:
            raise NotImplementedError

        acc = metrics.accuracy_score(query_labels, query_pred)
        return acc

    def distribute_model(self, num_gpus):
        """
        Distribte the backbone over multiple GPUs
        """
        if num_gpus > 1:
            self.encoder.cuda(0)
            self.encoder = torch.nn.DataParallel(self.encoder, device_ids=[i for i in range(num_gpus)])


class SupCEResNet(nn.Module):
    """encoder + classifier"""
    def __init__(self, name='resnet50', num_classes=10):
        super(SupCEResNet, self).__init__()
        model_fun, dim_in = model_dict[name]
        self.encoder = model_fun()
        self.fc = nn.Linear(dim_in, num_classes)

    def forward(self, x):
        return self.fc(self.encoder(x))
    

class LinearClassifier(nn.Module):
    """Linear classifier"""
    def __init__(self, name='resnet50', num_classes=10):
        super(LinearClassifier, self).__init__()
        _, feat_dim = model_dict[name]
        self.fc = nn.Linear(feat_dim, num_classes)

    def forward(self, features):
        return self.fc(features)
