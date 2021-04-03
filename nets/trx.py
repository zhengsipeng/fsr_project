import math
import torch
import torch.nn as nn
import torchvision.models as models
from itertools import combinations 
from .utils import freeze_all, freeze_bn, initialize_linear, initialize_3d
from .base_net import BaseNet


class PositionalEncoding(nn.Module):
    "Implement the PE function."
    def __init__(self, d_model, dropout, max_len=5000, pe_scale_factor=0.1):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.pe_scale_factor = pe_scale_factor
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term) * self.pe_scale_factor
        pe[:, 1::2] = torch.cos(position * div_term) * self.pe_scale_factor
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
                          
    def forward(self, x):
       x = x + self.pe[:, :x.size(1)].requires_grad_(requires_grad=False)
       return self.dropout(x)


class TemporalCrossTransformer(nn.Module):
    def __init__(self, args, temporal_set_size=3):
        super(TemporalCrossTransformer, self).__init__()
       
        self.args = args
        self.temporal_set_size = temporal_set_size

        max_len = int(self.args.sequence_length * 1.5)
        self.pe = PositionalEncoding(self.args.trans_linear_in_dim, self.args.trans_dropout, max_len=max_len)

        self.k_linear = nn.Linear(self.args.trans_linear_in_dim * temporal_set_size, self.args.trans_linear_out_dim)
        self.v_linear = nn.Linear(self.args.trans_linear_in_dim * temporal_set_size, self.args.trans_linear_out_dim)

        self.norm_k = nn.LayerNorm(self.args.trans_linear_out_dim)
        self.norm_v = nn.LayerNorm(self.args.trans_linear_out_dim)
        
        self.class_softmax = torch.nn.Softmax(dim=1)
        
        # generate all tuples
        frame_idxs = [i for i in range(self.args.sequence_length)]
        frame_combinations = combinations(frame_idxs, temporal_set_size)
        self.tuples = [torch.tensor(comb).cuda() for comb in frame_combinations]
        self.tuples_len = len(self.tuples) 
    
    def forward(self, shot, query, support_labels):
        n_query = query.shape[0]
        n_support = shot.shape[0]
        
        # static pe
        shot = self.pe(shot)
        query = self.pe(query)

        # construct new query and support set made of tuples of images after pe
        s = [torch.index_select(shot, -2, p).reshape(n_support, -1) for p in self.tuples]
        q = [torch.index_select(query, -2, p).reshape(n_query, -1) for p in self.tuples]
        shot = torch.stack(s, dim=-2)
        query = torch.stack(q, dim=-2)

        # apply linear maps
        shot_ks = self.k_linear(shot)
        query_ks = self.k_linear(query)
        shot_vs = self.v_linear(shot)
        query_vs = self.v_linear(query)
        
        # apply norms where necessary
        mh_shot_ks = self.norm_k(shot_ks)
        mh_query_ks = self.norm_k(query_ks)
        mh_shot_vs = shot_vs
        mh_query_vs = query_vs
        
        unique_labels = torch.unique(support_labels)

        # init tensor to hold distances between every support tuple and every query tuple
        all_distances_tensor = torch.zeros(n_query, self.args.way).cuda()

        for label_idx, c in enumerate(unique_labels):
            # select keys and values for just this class
            class_k = torch.index_select(mh_shot_ks, 0, self._extract_class_indices(support_labels, c))
            class_v = torch.index_select(mh_shot_vs, 0, self._extract_class_indices(support_labels, c))
            k_bs = class_k.shape[0]

            class_scores = torch.matmul(mh_query_ks.unsqueeze(1), class_k.transpose(-2,-1)) / math.sqrt(self.args.trans_linear_out_dim)

            # reshape etc. to apply a softmax for each query tuple
            class_scores = class_scores.permute(0,2,1,3)
            class_scores = class_scores.reshape(n_query, self.tuples_len, -1)
            class_scores = [self.class_softmax(class_scores[i]) for i in range(n_query)]
            class_scores = torch.cat(class_scores)
            class_scores = class_scores.reshape(n_query, self.tuples_len, -1, self.tuples_len)
            class_scores = class_scores.permute(0,2,1,3)
            
            # get query specific class prototype         
            query_prototype = torch.matmul(class_scores, class_v)
            query_prototype = torch.sum(query_prototype, dim=1)
            
            # calculate distances from query to query-specific class prototypes
            diff = mh_query_vs - query_prototype
            norm_sq = torch.norm(diff, dim=[-2,-1])**2
            distance = torch.div(norm_sq, self.tuples_len)
            
            # multiply by -1 to get logits
            distance = distance * -1
            c_idx = c.long()
            all_distances_tensor[:,c_idx] = distance
        
        return all_distances_tensor

    @staticmethod
    def _extract_class_indices(labels, which_class):
        """
        Helper method to extract the indices of elements which have the specified label.
        :param labels: (torch.tensor) Labels of the support set.
        :param which_class: Label for which indices are extracted.
        :return: (torch.tensor) Indices in the form of a mask that indicate the locations of the specified label.
        """
        class_mask = torch.eq(labels, which_class)  # binary mask of labels equal to which_class
        class_mask_indices = torch.nonzero(class_mask)  # indices of labels equal to which class
        return torch.reshape(class_mask_indices, (-1,))  # reshape to be a 1D vector
        

class TRX(BaseNet):
    """
    Standard Resnet connected to a Temporal Cross Transformer.
    """
    def __init__(self, args, way=5, shot=1, query=5):
        super(TRX, self).__init__(args)
        self.metric = args.metric
        self.transformers = nn.ModuleList([TemporalCrossTransformer(args, s) for s in args.temp_set]) 

    def forward(self, shot, query, labels, aux):
        support_labels = torch.arange(self.way).repeat(self.shot).to(device)
        
        x = torch.cat((shot, query), dim=0)
        b, d, c, h, w = x.shape  # way*(shot+query), num_frames, channel, height, weight
        x = x.view(b * d, c, h, w)

        if self.multi_modal:
            mm_x= [x]
            if self.use_depth:
                depth = aux['depth'].view(b * d, c, h ,w)
                mm_x = mm_x + [depth]
            if self.use_pose:
                pose = aux['pose'].view(b * d, c, h ,w)
                mm_x = mm_x + [pose]
            if self.use_flow:
                flow = aux['flow'].view(b * d, c, h ,w)
                mm_x = mm_x + [flow]
            x = mm_x
        
        # encoder
        x = self.encoder(x)
        x = torch.cat(x, dim=1).squeeze() 
        x = x.view(b, d, -1)

        shot, query = x[:shot.size(0)], x[shot.size(0):]
        dim = int(shot.shape[-1])
        #print(shot.shape)
        #assert 1==0
        shot = shot.view(-1, self.sequence_length, self.last_dim)
        query = query.view(-1, self.sequence_length, self.last_dim)
        all_logits = [t(shot, query, support_labels) for t in self.transformers] 
        all_logits = torch.stack(all_logits, dim=-1)
        sample_logits = all_logits   # [way*num_query, way]
        logits = torch.norm(sample_logits, dim=[-1]) * -1
    
        return logits


if __name__ == "__main__":
    class ArgsObject(object):
        def __init__(self):
            self.trans_linear_in_dim = 512
            self.trans_linear_out_dim = 128

            self.way = 5
            self.shot = 1
            self.query_per_class = 5
            self.trans_dropout = 0.1
            self.sequence_length = 8 
            self.img_size = 84
            self.backbone = "resnet18"
            self.num_gpus = 1
            self.temp_set = [2,3]

    args = ArgsObject()
    torch.manual_seed(0)
    device = 'cuda:0'
    model = TRX(args).to(device)
    
    support_imgs = torch.rand(args.way * args.shot * args.sequence_length,3, args.img_size, args.img_size).to(device)
    query_imgs = torch.rand(args.way * args.query_per_class * args.sequence_length ,3, args.img_size, args.img_size).to(device)
    support_labels = torch.tensor([0,1,2,3,4]).to(device)

    print("Support images input shape: {}".format(support_imgs.shape))
    print("query images input shape: {}".format(query_imgs.shape))
    print("Support labels input shape: {}".format(support_imgs.shape))

    out = model(support_imgs, support_labels, query_imgs)

    print("TRX returns the distances from each query to each class prototype.  Use these as logits.  Shape: {}".format(out['logits'].shape))

