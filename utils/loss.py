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
        pos_num = pos_onehot.sum(1)  # b, 1
        #print(pos_onehot)
        #print(torch.sum(pos_onehot, dim=1))
        
        # cross entropy loss for action classification
        ce_loss = self.criterion_ce(logits, labels)

        # MSE loss for spatial coordinates
        sp_dist = torch.sqrt(((st_locs - st_locs_back)**2).sum(2))  # b, p_num
        log_softmax_sp = -pos_onehot * torch.log(F.softmax(sp_dist, dim=1))  # b, p_num
        infonce_sp_loss = torch.div(log_softmax_sp.sum(1), pos_num+1e-4).mean()

        # InfoNCE loss for contrastive leraning
        softmax_12 = F.softmax(p_sim_12, dim=1)
        softmax_21 = F.softmax(p_sim_21, dim=1)
        log_softmax = -pos_onehot * torch.log(softmax_12 * softmax_21) # -1/pos_num x sum(log(pos_instance)) 
        info_nce_loss = torch.div(log_softmax.sum(1), pos_num+1e-4).mean()
        #print(infonce_sp_loss, info_nce_loss)
        losses = self.sigma_ce*ce_loss + self.sigma_sp*infonce_sp_loss + self.sigma_feat*info_nce_loss

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