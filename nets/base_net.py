"""
Meta-learning method sets including MatchNet, RelatioNet, ProtoNet
"""

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
from .resnet import resnet18, resnet34, resnet50
from .utils import freeze_all
from sklearn import metrics
from sklearn.svm import SVC, LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


class BaseNet(nn.Module):
    def __init__(self, args):
        super(BaseNet, self).__init__()
        self.way = args.way
        self.shot = args.shot
        self.query = args.query
        self.sequence_length = args.sequence_length
        self.bsz = args.batch_size
        self.num_classes = 64+12
        
        # multi-modal 
        self.use_depth = args.use_depth
        self.use_pose = args.use_pose
        self.use_flow = args.use_flow
        self.num_modal = 1 + int(self.use_depth)+int(self.use_pose)+int(self.use_flow)
        self.fusion = args.fusion
        self.sharing = args.sharing
        self.bn_threshold = args.bn_threshold
        self.multi_modal = False
        if self.use_depth or self.use_pose or self.use_flow:
            self.multi_modal = True

        self.name = args.backbone
        if self.name == 'resnet18' or self.name == 'resnet34':
            self.dim_in = 512
        else:
            self.dim_in = 2048

        self.classifier = args.classifier
        self.freeze_all = args.freeze_all
        self.method = args.method
        if self.method == 'proto':
            self.classifier = 'Proto'
        self.encoder = self.build_backbone()
        self.action_classifier = nn.Linear(self.dim_in, self.num_classes)

    def build_backbone(self):
        if self.multi_modal:
            num_parallel = 1+int(self.use_depth)+int(self.use_pose)+int(self.use_flow)
            if self.backbone == "resnet18":
                resnet = resnet18(pretrained=True, parallel=num_parallel, num_parallel=num_parallel, 
                                  bn_threshold=self.bn_threshold, fusion=self.fusion) 
            elif self.backbone == "resnet34":
                resnet = resnet34(pretrained=True, parallel=num_parallel, num_parallel=num_parallel, 
                                  bn_threshold=self.bn_threshold, fusion=self.fusion)
            elif self.backbone == "resnet50":
                resnet = resnet50(pretrained=True, parallel=num_parallel, num_parallel=num_parallel, 
                                  bn_threshold=self.bn_threshold, fusion=self.fusion)
        else:
            if self.name == "resnet18":
                resnet = models.resnet18(pretrained=True)  
            elif self.name == "resnet34":
                resnet = models.resnet34(pretrained=True)
            elif self.name == "resnet50":
                resnet = models.resnet50(pretrained=True)
            
        if self.method == 'cycle':
            resnet = nn.Sequential(
                resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool,
                resnet.layer1, resnet.layer2, resnet.layer3, resnet.layer4)
        else:
            resnet = nn.Sequential(
                resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool,
                resnet.layer1, resnet.layer2, resnet.layer3, resnet.layer4,
                resnet.avgpool)

        #if self.freeze_all:
        #    resnet.apply(freeze_all)

        return resnet

    def simple_classifier(self, classifier, support_feats, query_feats, support_labels, query_labels):
        if classifier == 'LR' or classifier == 'SVM':
            support_feats = support_feats.cpu().detach().numpy()
            query_feats = query_feats.cpu().detach().numpy()
            support_labels = support_labels.cpu().detach().numpy()
            query_labels = query_labels.cpu().detach().numpy()
        
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

        if classifier == 'LR' or classifier == 'SVM':
            query_pred = torch.from_numpy(query_pred)
        
        return query_pred

    def multi_modal_aux(self, x, aux, is_pretrain):
        if is_pretrain:
            num = self.bsz*2*self.sequence_length
        else:
            num = self.b*self.sequence_length

        _, _, c, h, w = x.shape

        if self.multi_modal:
            mm_x= [x]
            if self.use_depth:
                depth = aux['depth'].view(num, c, h ,w)
                mm_x = mm_x + [depth]
            if self.use_pose:
                pose = aux['pose'].view(num, c, h ,w)
                mm_x = mm_x + [pose]
            if self.use_flow:
                flow = aux['flow'].view(num, c, h ,w)
                mm_x = mm_x + [flow]
            x = mm_x
            #x = torch.cat(x, dim=1).squeeze()
        return x

    def distribute_model(self, num_gpus):
        """
        Distribte the backbone over multiple GPUs
        """
        if num_gpus > 1:
            self.encoder.cuda(0)
            self.encoder = torch.nn.DataParallel(self.encoder, device_ids=[i for i in range(num_gpus)])


def Proto(support, support_labels, query, way, shot):
    """Protonet classifier"""
    nc = support.shape[-1]
    support = support.reshape(-1, 1, way, shot, nc)
    support = support.mean(dim=3)
    batch_size = support.shape[0]
    query = query.reshape(batch_size, -1, 1, nc)  
    logits = -((query - support)**2).sum(-1)
    pred = torch.argmax(logits, dim=-1)
    pred = pred.reshape(-1).unsqueeze(-1)
    return logits, pred 


def NN(support, support_labels, query):
    """nearest classifier"""
    support = support.T
    support = support.unsqueeze(0)
    query = query.unsqueeze(2)

    diff = torch.mul(query - support, query - support)
    distance = diff.sum(1)
    min_idx = torch.argmin(distance, dim=1)
    pred = support[min_idx] # [support_labels[idx] for idx in min_idx]
    return pred


def Cosine(support, support_labels, query):
    """Cosine classifier"""
    support_norm = torch.norm(support, p=2, dim=1, keepdim=True)
    support = support / support_norm
    query_norm = torch.norm(query, p=2, dim=1, keepdim=True)
    query = query / query_norm

    cosine_distance = torch.mm(query, support.T)
    max_idx = torch.argmax(cosine_distance, dim=1)
    pred = support[max_idx] 
    return pred