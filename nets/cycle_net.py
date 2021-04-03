import math
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from .utils import average_pool
from .base_net import BaseNet


class CycleNet(BaseNet):
    def __init__(self, args, device):
        super(CycleNet, self).__init__(args)
        self.patch_scales = [2, 4]

        if not self.sharing:
            self.encoder2 = self.build_backbone()  # backbone for modality 2
            if self.use_depth and self.use_pose and self.use_flow:
                self.encoder3 = self.build_backbone()
        
        # --------------------
        # pretraining setting
        # --------------------
        # average pool for multi-level patch
        self.p_avg_pool = average_pool
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.m_st_locs = self.get_m_st_loc(patch_scales=[1]+self.patch_scales, device=device) # 7*7 + 4*4 + 2*2 = 49+16+4=67

    def get_m_st_loc(self, patch_scales, T=8, H=7, W=7, stride=None, device=None):
        """ 
        generate multi_scale spatial-temporal locations
        """
        m_st_locs = []
        for scale in patch_scales:
            stride = scale
            sp_idxs = []
            for t in range(T):
                for h in range(math.ceil(H/stride)):
                    for w in range(math.ceil(W/stride)):
                        # we decrease the temporal weight
                        sp_idxs.append([t/2., stride*h+scale/2., stride*w+scale/2])
            m_st_locs += sp_idxs
        m_st_locs = torch.Tensor(m_st_locs).to(device)
        return m_st_locs

    def forward(self, datas, aux, labels=None, is_pretrain=True):
        """
        datas: batchsize, s, t, C, H, W
        aux: "depth", "pose", "flow",   batchsize, 2, t, C, H, W
        """
        if is_pretrain:
            """
            when pretraining, we use cycle consistency and contrastive learning to pretrain the backbone
            """
            b, _, t, c, h, w = datas.shape  # d: single or double
            x = datas.view(b*2*t, c, h, w)
            x = self.multi_modal_aux(x, aux, is_pretrain)
            x = self.encoder(x)  # b*2*t, c, 7, 7
            
            # action classification
            _x = self.avgpool(x).squeeze()  # b*s*t, c
            _x = _x.reshape(b, 2, t, self.last_dim)
            _x = _x[:, 1, :, :].mean(1)  # b, t, c -> b, c
            logits = self.action_classifier(_x)

            _x = torch.transpose(x.reshape(b, 2, t, self.last_dim, 7, 7), 2, 3).reshape(b, 2, self.last_dim, -1)  # b, 2, c, t, 7, 7
            m_scale_feats = [_x]  # multi-scale features
            for p_scale in self.patch_scales:
                # the first avg pool layer you should notice
                _feats1 = self.p_avg_pool(x, 
                                        kernel_size=(p_scale, p_scale), 
                                        padding=(p_scale-7%p_scale, p_scale-7%p_scale), 
                                        stride=(p_scale, p_scale))  # b*2*t, c, ceil(7/p_scale), ceil(7/p_scale)
                _feats1 = _feats1.reshape(b, 2, t, self.last_dim, math.ceil(7/p_scale), math.ceil(7/p_scale))
                _feats1 = torch.transpose(_feats1, 2, 3).reshape(b, 2, self.last_dim, -1)  
                m_scale_feats.append(_feats1)
            
            m_scale_feats = torch.cat(m_scale_feats, dim=3)  # b, 2, c, 8*49+8*16+8*4 = 552
            p_num = m_scale_feats.shape[-1]
            
            # TODO
            # 1. 552 patch is too much, only consider the spatial when trace back
            # 2. add temporal weight on feature similarity (higher similar with closer temporal distance) 
            p_feats1 = torch.transpose(m_scale_feats[:, 0, :, :], 1, 2)  # b, c, 552 -> b, 552, c
            p_feats2 = torch.transpose(m_scale_feats[:, 1, :, :], 1, 2)  # b, c, p_num -> b, p_num, c

            # normalize the feature
            norm_f1 = (p_feats1*p_feats1).sum(2).sqrt().unsqueeze(2)
            norm_f2 = (p_feats2*p_feats2).sum(2).sqrt().unsqueeze(2)
            norm_f1 = torch.div(p_feats1, norm_f1)
            norm_f2 = torch.transpose(torch.div(p_feats2, norm_f2), 1, 2)
            
            # consine similarity
            _norm_f2 = torch.transpose(norm_f2, 1, 2)  # b, p_num, c -> b, c, p_num
            p_sim_12 = torch.einsum('bpc,bcn->bpn', [norm_f1, norm_f2])*1   # b, p_num, p_num

            # trace from 1 -> 2
            maxids_12 = torch.argmax(p_sim_12, dim=2)  # b, p_num
            traceids_12 = maxids_12  # b, p_num
            # trace back 2 -> 1
            p_sim_21 = torch.transpose(p_sim_12, 1, 2)
            maxids_21 = torch.argmax(p_sim_21, dim=2)  # b, p_num
            traceids_21 = maxids_21.gather(dim=1, index=traceids_12)  # b, p_num 
            
            #p_sim_12 = F.softmax(p_sim_12, dim=2)  # type3
            p_sim_12, _ = torch.max(p_sim_12, dim=2)  # b, p_num, p_num -> b, p_num 
            p_sim_21 = torch.gather(p_sim_21, dim=2, index=traceids_21.unsqueeze(2)).squeeze(-1)
            
   
            # spatial-temporal similarity for MSE loss
            st_locs = self.m_st_locs.unsqueeze(0).repeat(b, 1, 1)  # p_num, 3 -> b, p_num, 3
            st_locs_back = torch.index_select(st_locs.view(b*p_num, 3), 0, traceids_21.view(-1))
            st_locs_back = st_locs_back.view(b, p_num, 3)

            # initially p_num cycle pairs
            # then we choose the positive cycles with sim > threshold
            _norm_f1 = torch.transpose(norm_f1, 1, 2)
            p_sim_11 = torch.einsum('bpc,bcn->bpn', [norm_f1, _norm_f1])   # b, p_num, p_num
            pos_sim = torch.gather(p_sim_11, dim=2, index=traceids_21.unsqueeze(2)).squeeze(-1)  # b, p_num

            return logits, st_locs, st_locs_back, p_sim_12, p_sim_21, pos_sim
        else:
            """
            for meta-testing, we simply use linear classifier by sklearn
            """
            x = datas
            b, t, c, h, w = x.shape # way*(shot+query), num_frame, c, h, w
            x = x.view(b*t, c, h, w)
            x = self.multi_modal_aux(x, aux, is_pretrain)
            x = self.encoder(x)
            x = self.res_avgpool(x).squeeze()
            x = x.reshape(b, t, self.last_dim).mean(1)
            
            pivot = self.way * self.shot
            support_feats, query_feats = x[:pivot], x[pivot:] 
            support_labels = labels[:pivot]
            query_labels = labels[pivot:]

            query_pred = self.simple_classifier(self.classifier, support_feats, query_feats, support_labels, query_labels)
            
            return query_pred
    

#p_feats1_back = torch.index_select(p_feats1.reshape(b*p_num, self.last_dim), 0, trace2_idxs.reshape(-1))
#p_feats1_back = p_feats1_back.view(b, p_num, self.last_dim)  # p, p_num, 3
#p_feats1_back = torch.transpose(p_feats1.gather(dim=1, index=trace2_idxs), 1, 2)  # b, p_num, c
#p_back_sim = torch.einsum('bpc,bcn->bpn', [p_feats1_back, p_feats1])  # b, p_num, p_num
#print(p_back_sim)
#p_back_sim, _ = torch.max(F.softmax(p_back_sim, dim=2), dim=2)  # b, p_num
#p_back_sim = p_back_sim.gather(dim=1, index=trace2_idxs)  # b, p_num -> b, p_num

#_p_feats1 = p_feats1.repeat(1, 1, p_num).reshape(b*p_num*p_num, c)  # b, p_num, p_num*c -> b*p_num*p_num, c
#_p_feats2 = p_feats2.repeat(1, p_num, 1).reshape(b*p_num*p_num, c)  
#_p_feats1 = p_feats1.expand(b, p_num, p_num*self.last_dim)
#_p_feats1 = _p_feats1.reshape(b*p_num*p_num, self.last_dim)
#_p_feats2 = p_feats2.expand(b, p_num*p_num, self.last_dim).reshape(b*p_num*p_num, self.last_dim)  
#p_sim_12 = torch.cosine_similarity(_p_feats1, _p_feats2, dim=1).reshape(b, p_num, p_num) 
#p_sim_12, _ = torch.max(p_sim_12, dim=2)  # b, p_num, p_num -> b, p_num  




