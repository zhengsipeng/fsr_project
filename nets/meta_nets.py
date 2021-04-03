import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from .base_net import BaseNet


class MetaProtos(BaseNet):
    def __init__(self, args, head='mlp', feat_dim=512, num_gpus=4):
        super(MetaProtos, self).__init__(args)

        if head == 'linear':
            self.head = nn.Linear(self.dim_in, feat_dim)
        elif head == 'mlp':
            self.head = nn.Sequential(
                nn.Linear(self.dim_in, self.dim_in),
                nn.ReLU(inplace=False),
                nn.Linear(self.dim_in, feat_dim)
            )
        else:
            raise NotImplementedError(
                'head not supported: {}'.format(head))
        self.fc = nn.Sequential(
                nn.Linear(2048, 256),
                nn.ReLU(inplace=False),
                nn.Linear(256, 5)
            )
        self.use_ce = args.use_ce
        
    def forward(self, x, aux=None, labels=None, is_eval=False):
        #TODO: multi-modality fusion
        b, t, c, h, w = x.shape # way*shot+way*query, num_frame, c, h, w
        
        x = x.reshape(b*t, c, h, w)
        x = self.encoder(x).reshape(b, t, -1).mean(1)
     
        pivot = self.way * self.shot
        support_feats, query_feats = x[:pivot], x[pivot:] 
        support_labels = labels[:pivot]
        query_labels = labels[pivot:]
        logits = self.simple_classifier(self.classifier, support_feats, query_feats, support_labels, query_labels)
        
        return logits

    