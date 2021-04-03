import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from .base_net import BaseNet


class SupConResNet(BaseNet):
    """backbone + projection head"""
    def __init__(self, args, head='mlp', feat_dim=512, num_gpus=4):
        super(SupConResNet, self).__init__(args)

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

        self.use_ce = args.use_ce

    def forward(self, x, aux=None, labels=None, is_eval=False):
        if not is_eval:
            b2, t, c, h, w = x.shape
            x = x.reshape(b2*t, c, h, w)           
            bsz = int(b2/2)
            
            feat = self.encoder(x)
            feat = feat.reshape(b2, t, -1).mean(1)  # b2, c
            logits = self.action_classifier(feat[:bsz])

            feat = torch.flatten(feat, 1)
            feat = self.head(feat)
            #print(feat.shape)
            feat = F.normalize(feat, dim=1)
            f1, f2 = torch.split(feat, [bsz, bsz], dim=0)
            feat = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1)  # bsz, 2, c

            return logits, feat
            
        else:  # evaluation or meta-train
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
