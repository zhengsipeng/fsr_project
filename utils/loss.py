import torch
import torch.nn as nn
import torch.nn.functional as F

# =================================================
# Combinational Losses for Consistency Learning
# the combination losses for consistency pretraining consist of:
# (1) cross entropy loss for action classification (CE loss)
# (2) spatial cycle consistency (MSE loss)
# (3) feature similarity of cycle consistency (consistency loss)

# logits&labels: batchsize, 64 & batchsize, 1
# st_locs&st_locs_back: batchsize, p_num
# log_p_sim_12&log_p_sim_21: batchsize, p_num
# pos_onehot: batchsize, p_num
# =================================================
class PatchCycleLoss(nn.Module):
    def __init__(self, args):
        super(PatchCycleLoss, self).__init__()
        self.sigma_ce = args.sigma_ce
        self.sigma_sp = args.sigma_sp
        self.sigma_feat = args.sigma_feat
        self.criterion_ce = nn.CrossEntropyLoss(reduction="mean")
        self.criterion_mse = nn.MSELoss()

    def forward(self, logits, labels, st_locs, st_locs_back, p_sim_12, p_sim_21, pos_onehot):
        """
        pos_onehot: batchsize, p_num
        p_sim_12 & p_sim_21, batchsize, p_num
        """
        p_num = pos_onehot.shape[1]
        pos_num = pos_onehot.sum(1)  # b, 1
    
        # cross entropy loss for action classification
        ce_loss = self.criterion_ce(logits, labels)*self.sigma_ce

        # MSE loss for spatial coordinates
        sp_dist = torch.sqrt(((st_locs - st_locs_back)**2).sum(2))  # b, p_num

        log_softmax_sp = -pos_onehot * torch.log(F.softmax(sp_dist, dim=1))  # b, p_num
        infonce_sp_loss = torch.div(log_softmax_sp.sum(1), pos_num+1e-4).mean()*self.sigma_sp

        # InfoNCE loss for contrastive leraning
        # type1
        prob = p_sim_12 * p_sim_21*10  # b, p_num
        #print(prob)
        prob = F.softmax(prob, dim=1)
        #print(prob[0])
        # normalization
        mean = prob.mean(dim=1).unsqueeze(1)
        stdv = torch.sqrt(((prob-mean)**2).sum(dim=1)/(p_num-1)).unsqueeze(1)
        prob = ((prob-mean)/stdv)
        softmax_prob = F.softmax(prob, dim=1)
        log_softmax = -pos_onehot * torch.log(softmax_prob)
        #print(prob[0])
        #assert 1==0
        # type2
        #softmax_12 = F.softmax(p_sim_12, dim=1)
        #softmax_21 = F.softmax(p_sim_21, dim=1)        
        #log_softmax = -pos_onehot * torch.log(softmax_12 * softmax_21) # -1/pos_num x sum(log(pos_instance)) 
        

        '''
        # type3
        mean = p_sim_12.mean(dim=1).unsqueeze(1)
        stdv = torch.sqrt(((p_sim_12-mean)**2).sum(dim=1)/(p_num-1)).unsqueeze(1)
        p_sim_12 = (p_sim_12-mean)/stdv
        mean = p_sim_21.mean(dim=1).unsqueeze(1)
        stdv = torch.sqrt(((p_sim_21-mean)**2).sum(dim=1)/(p_num-1)).unsqueeze(1)
        p_sim_21 = (p_sim_21-mean)/stdv
        log_softmax = -pos_onehot * torch.log(p_sim_12 * p_sim_21)
        print(p_sim_12[0])
        print(p_sim_21[0])
        '''
        
        #print(mean[0])
        #print(stdv[0])
        #assert 1==0
        
        # type3
        #print(log_softmax[0])
        info_nce_loss = torch.div(log_softmax.sum(1), pos_num+1e-4).mean()*self.sigma_feat
        
        #print(infonce_sp_loss, info_nce_loss)
        losses = ce_loss + infonce_sp_loss + info_nce_loss

        #print(losses)
        #print(ce_loss, infonce_sp_loss, info_nce_loss)
        #losses = ce_loss
        #assert 1==0
        return losses, ce_loss, infonce_sp_loss, info_nce_loss


# =============================
# Basic TRX Cross Entropy Loss
# =============================
def cross_entropy_trx(logits, labels, device):
    """
    Compute the classification loss for TRX.
    logits: [1, way * num_query, way]
    labels: [way * num_query]
    """
    size = logits.size()
    num_samples = torch.tensor([1], dtype=torch.float, device=device, requires_grad=False)
    log_py = torch.empty(size=(1, size[0]), dtype=torch.float, device=device)
    log_py[0] = -F.cross_entropy(logits, labels, reduction='none')
    score = torch.logsumexp(log_py, dim=0) - torch.log(num_samples)
    return -torch.sum(score, dim=0) / 16


# =========================
# Basic Cross Entropy Loss
# =========================
def cross_entropy_loss(args, test_logits_sample, test_labels, device):
    """
    Compute the classification loss.
    """
    # test_logits_sample: [1, way * num_query, way]
    # test_labels: [way * num_query]
    
    size = test_logits_sample.size()
    sample_count = size[0]  # scalar for the loop counter
    num_samples = torch.tensor([sample_count], dtype=torch.float, device=device, requires_grad=False)

    log_py = torch.empty(size=(size[0], size[1]), dtype=torch.float, device=device)
    if args.model == 'trx':
        for sample in range(sample_count):
            log_py[sample] = -F.cross_entropy(test_logits_sample[sample], test_labels, reduction='none')
            score = torch.logsumexp(log_py, dim=0) - torch.log(num_samples)
        return -torch.sum(score, dim=0)
    else:
        return F.cross_entropy(test_logits_sample[0], test_labels)
    

"""
#_pos_onehot = pos_onehot.unsqueeze(2).repeat(1, 1, 3)
#masked_st_locs = (st_locs*_pos_onehot).reshape(-1, 3)
#masked_st_locs_back = (st_locs_back*_pos_onehot).reshape(-1, 3)
#sp_loss = self.criterion_mse(masked_st_locs, masked_st_locs_back)  # only calculate the positive pairs
"""


class SupConLoss(nn.Module):
    def __init__(self, args, contrast_mode='all', base_temperature=0.07):
        super(SupConLoss, self).__init__()
        self.args = args
        self.use_ce = args.use_ce
        self.cross_entropy = nn.CrossEntropyLoss(reduction="mean")
        self.temperature = args.temp
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature
    
    def forward(self, features, logits, labels, loss_type=''):
        """
        features need to be [batchsize, n_views, ...] at least 3 dimenstions are required
        """
        device = torch.device('cuda') if features.is_cuda else torch.device('cpu')
        batch_size = features.shape[0]
        
        #print(features.shape, logits.shape, labels.shape)
        #assert 1==0
        if loss_type == 'SimCLR':
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif loss_type == 'SupCon':
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device)  # bsz, bsz

        contrast_count = features.shape[1]  # n_views
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)  # bsz*2, c

        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count  # 2
        
        # compute logits
        anchor_dot_contrast = torch.div(torch.matmul(anchor_feature, contrast_feature.T), self.temperature)
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()  # bsz*2, bsz*2

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )  # mask i=i
        mask = mask * logits_mask

        # compute log prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))  # log(x/y) = logx - logy

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        # loss 
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        contrast_loss = loss.view(anchor_count, batch_size).mean()

        if self.use_ce:
            ce_loss = self.cross_entropy(logits, labels)#*self.sigma_ce
            loss = self.sigma_contrast_loss + self.sigma_ce*ce_loss
            return loss, ce_loss, contrast_loss
        else:
            return contrast_loss, 0, contrast_loss
        
        
