import math
import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F


class Memory():
    """Memory module"""
    def __init__(self, batchs_size, key_dim, mem_depth, num_mem_slots, vocab_size,
               choose_k=256, alpha=0.1, correct_in_top=1, age_noise=8.0,
               var_cache_device='', nn_device=''):
        self.batch_size = batch_size
        self.key_dim = key_dim
        self.mem_depth = mem_depth
        self.num_mem_slots = num_mem_slots
        self.vocab_size = vocab_size
        self.choose_k = min(choose_k, num_mem_slots)
        self.alpha =alpha
        self.correct_in_top = correct_in_top
        self.age_noise = age_noise
        self.var_cache_device = var_cache_device  # Variables are cached here.
        self.nn_device = nn_device  # Device to perform nearest neighbour matmul.

        self.num_centers = mem_depth

        caching_device = var_cache_device
        self.update_memory = torch.Tensor(True)
        self.external_centers = self.truncated_normal_([self.num_centers, self.key_dim])
        self.mem_keys = self.truncated_normal_([self.num_mem_slots, self.mem_depth, self.key_dim])
        self.mem_hashed_keys = self.truncated_normal_([self.num_mem_slots, self.key_dim])
        self.mem_vals = torch.zeros([self.num_mem_slots], dtype=torch.int)
        self.mem_age = torch.zeros([self.num_mem_slots], dtype=torch.float32)
        self.recent_idx = torch.zeros([self.vocab_size], dtype=torch.int)
    
        # memory layers
        self.memory_conv2d = torch.nn.Conv2d(self.batch_size, self.key_dim, (1, 1))
        cell_size = 512
        self.gru_cell = torch.nn.GRUCell(cell_size, cell_size)
        self.linear_layer = nn.Linear(self.batch_size, self.key_dim)
        self.linear_layer2 = nn.Linear(self.batch_size, self.key_dim)

    def truncated_normal_(self, size, mean=0, std=0.09):
        tensor = torch.randn(size)
        with torch.no_grad():
            size = tensor.shape
            tmp = tensor.new_empty(size+(4,)).normal_()
            valid = (tmp < 2) & (tmp > -2)
            ind = valid.max(-1, keepdim=True)[1]
            tensor.data.copy_(tmp.gather(-1, ind).squeeze(-1))
            tensor.data.mul_(std).add_(mean)
            return tensor

    def get(self):
        return self.mem_keys, self.mem_vals, self.mem_age, self.recent_idx
    
    def set(self, k, v, a, r=None):
        return None
    
    def clear(self):
        self.mem_keys = self.truncated_normal_([self.num_mem_slots, self.mem_depth, self.key_dim])
        self.mem_hashed_keys = self.truncated_normal_([self.num_mem_slots, self.key_dim])
        self.mem_vals = torch.zeros([self.num_mem_slots], dtype=torch.int)
        self.mem_age = torch.zeros([self.num_mem_slots], dtype=torch.float32)
        self.recent_idx = torch.zeros([self.vocab_size], dtype=torch.int)

    def get_hint_pool_idxs(self, normalized_query):
        # look up in large memory, no gradients
        similarities = torch.matmul(normalized_query.requires_grad_(requires_grad=False), self.mem_hashed_keys.transpose(1, 0))
        _, hint_pool_idxs = torch.topk(similarities.requires_grad_(requires_grad=False), k=self.choose_k)
        return hint_pool_idxs

    def make_update_op(self, upd_idxs, upd_keys, upd_center_keys, upd_vals,
                       batch_size, use_recent_idx, intended_output):
        """Function that creates all the update ops."""
        self.mem_age += torch.ones([self.num_mem_slots], dtype=torch.float32)

        self.mem_age[:, upd_idxs] = 0
        self.mem_hashed_keys[:, upd_idxs] = upd_keys
        self.mem_keys[:, upd_idxs] = upd_center_keys
        self.mem_vals[:, upd_idxs] = upd_vals

        if use_recent_idx:
            self.recent_idx[intended_output] = upd_idxs

    def query_C_mem(self, inputs, centers):
        """
        inputs: batch_size, x, seq_len, x, v
        centers: batch_size, x, num_centers, x, v
        """
        batch_size = inputs.shape[0]
        input_length = inputs.shape[1]
        num_centers = centers.shape[0]
        input_size = inpus.shape[2]
        centers = centers.view(-1, num_centers, input_size)
        a = torch.matmul(inputs, centers.transpose(1, 0))
        a *= torch.rsqrt(inputs.shape[2])
        a = torch.Softmax(a)
        #print(a[0, 200, 0], a[0, 200, 1], a[0, 200, 2], a[0, 200, 3], a[0, 200, 4])
        o = a.view(-1, input_length, num_centers, 1) * inputs.view(-1, input_length, 1, input_size) - \
            centers.view(-1, 1, num_centers, input_size)
        output = torch.sum(o, dim=1)
        return output

    def context_encoding(self, inputs):
        initial_state = torch.zeros(inputs.shape)
        enc_outputs, enc_state = self.gru_cell(inputs, initial_state)
        return enc_outputs

    def normalize(self, inputs, ssr=True, intra_norm=True, l2_norm=True):
        if ssr:
            inputs = torch.sign(inputs) * torch.sqrt(torch.abs(inputs) + 1e-12 )
        if intra_norm:
            inputs = F.normalize(inputs)
        return inputs

    def query(self, queries, intended_output, use_recent_idx=True, is_training=True):
        batch_size = queries.shape[0]
        output_given = intended_output is not None

        # prepare query for memory lookup
        queries = self.memory_conv2d(queries)

        if Flase:
            queries_with_context = self.context_encoding(queries)
        else:
            queries_with_context = queries
        
        queries_with_context = self.query_C_mem(queries_with_context, self.external_centers)
        # query_vec = torch.matmul(queries, self.query_proj)
        normalized_query = self.normalize(queries_with_context)
        if is_training:
            normalized_query = F.dropout(normalized_query, p=0.8)
        normalized_hashed_query = self.linear_layer(normalized_query)
        normalized_hashed_query = F.normalize(normalized_hashed_query)

        hint_pool_idxs = self.get_hint_pool_idxs(normalized_hashed_query)

        if output_given and use_recent_idx: # add at least one correct memory
            most_recent_hint = torch.gather(self.recent_idx, intended_output)
            hint_pool_idxs = torch.cat([hint_pool_idxs, torch.unsqueeze(most_recent_hint, 1)], dim=1)
        choose_k = hint_pool_idxs.shape[1]

        my_mem_hashed_keys = torch.gather(self.mem_hashed_keys, hint_pool_idxs)
        my_mem_hashed_keys = my_mem_hashed_keys.requires_grad_(requires_grad=False)

        hint_pool_mem_vals = torch.gather(self.mem_vals.requires_grad_(requires_grad=False), hint_pool_idxs)

        similarities = torch.matmul(torch.unsqueeze(normalized_hashed_query, 1), my_mem_hashed_keys)

        hint_pool_sims = torch.squeeze(similarities, 1)

        # calculate softmax mask on the top-k if requested
        # softmax temperature. Say we have K elements at dist x and one at (x+a)
        # Softmax of the last is e^tm(x+a)/Ke^tm*x + e^tm(x+a) = e^tm*a/K+e^tm*a
        # To mask that 20% we'd need to have e^tm*a ~= 0.2K, so tm=log(0.2K)/a

        softmax_temp = max(1., np.log(0.2*self.choose_k) / self.alpha)
        mask = F.Softmax(hint_pool_sims[:, :choose_k-1] * softmax_temp)

        # prepare hints from the teacher on hint pool
        teacher_hints = torch.abs(torch.squeeze(intended_output, 1) - hint_pool_mem_vals)
        teacher_hints = 1. - torch.minimum(1., teacher_hints)

        teacher_vals, teacher_hint_idxs = torch.topk(hint_pool_sims*teacher_hints, k=1)
        # zero-out teacher_vals if there are no hints
        teacher_vals *= 1- torch.equal(0, torch.sum(teacher_hints, dim=1, keepdim=True))
        neg_teacher_vals, _ = torch.topk(hint_pool_sims*(1 - teacher_hints), k=1)

        # loss based on triplet loss
        diff = neg_teacher_vals - teacher_vals
        teacher_loss = tf.nn.relu(diff+self.alpha)

        # prepare returned values
        nearest_neighbor = torch.int(torch.argmax(hint_pool_sims, 1))
        no_teacher_idxs = torch.gather(hint_pool_idxs.view(-1), 
                                       nearest_neighbor+choose_k*torch.range(batch_size))

        num_top_k_predicts = 5
        _, nearest_neighbors = torch.topk(hint_pool_sims[:, :choose_k-1], num_top_k_predicts)
        nearest_neighbors = torch.int(nearest_neighbors)
        
        batch_index = torch.repeat(torch.unsqueeze(torch.range(batch_size), 1), [1, num_top_k_predicts])

        index = torch.stack([batch_index, nearest_neighbors], 2)
        hint_pool_idxs_ = hint_pool_idxs.contiguous()
        inds = index.mv(torch.LongTensor(hint_pool_idxs_.stride()))
        nearest_idxs =torch.index_select(hint_pool_idxs_.view(-1), 0, inds)

        predicts = torch.gather(self.mem_vals, no_teacher_idxs.view(-1))
        top_k_predicts = torch.gather(self.mem_vals, nearest_idxs.view(-1)).view(-1, num_top_k_predicts)

        # prepare memory updates
        update_keys = normalized_hashed_query
        update_center_keys = queries_with_context
        update_vals =intended_output

        # bring back idxs to full memory
        teacher_idxs = torch.gather(hint_pool_idxs.view(-1), 
                                    teacher_hint_idxs[:, 0]+choose_k*torch.range(batch_size))
        fetched_idxs = teacher_idxs
        fetched_center_keys = torch.gather(self.mem_keys, fetched_idxs)
        fetched_vals = torch.gather(self.mem_vals, fetched_idxs)

        # do memory updates here
        fetched_center_keys_upd = queries_with_context + fetched_center_keys
        fetched_keys_upd = self.normalize(fetched_center_keys_upd)
        if is_training:
            fetched_keys_upd = F.dropout(fetched_keys_upd, p=0.8)
            fetched_keys_upd = self.linear_layer2(fetched_keys_upd)
            fetched_keys_upd = F.normalize(fetched_keys_upd, dim=1)
        
        # Randomize age a bit, e.g., to select different ones in parallel workers.
        mem_age_with_noise = self.mem_age + torch.Tensor(
                             [self.num_mem_slots]).uniform_(-self.age_noise, self.age_noise)

        _, oldest_idxs = torch.topk(mem_age_with_noise, k=batch_size, sorted=False)

        # we'll determine whether to do an update to memory based on whether
        # memory was queried correctly
        #sliced_hints = tf.slice(teacher_hints, [0, 0], [-1, self.correct_in_top])
        sliced_hints = teacher_hints.index_select(1, torch.range(self.correct_in_top))
        incorrect_memory_lookup = torch.equal(0.0, torch.sum(sliced_hints, dim=1))

        upd_idxs = torch.where(incorrect_memory_lookup, oldest_idxs, fetched_idxs)
        upd_keys = torch.where(incorrect_memory_lookup, update_keys, fetched_keys_upd)
        upd_center_keys = torch.where(incorrect_memory_lookup, update_center_keys, fetched_center_keys_upd)
        upd_vals = torch.where(incorrect_memory_lookup, update_vals, fetched_vals)
    
    self.make_update_op(upd_idxs, upd_keys, upd_center_keys, upd_vals,
                        batch_size, use_recent_idx, intended_output)
    
    #update_op = tf.cond(self.update_memory, make_update_op, tf.no_op)
    ps = []
    for i in range(5):
        pred_ = functional.one_hot(top_k_predicts[:, i], num_classes=self.vocab_size)
        on_value = 1. - i * 0.1
        pred = nn.init.constant_(pred_, on_value)
        ps.append(pred)
    logits = torch.add(pred)

    return predicts, mask, torch.mean(teacher_loss)


class CMN(nn.Module):
    def __init__(self, args):
        super(CNN_TRX, self).__init__()

        self.train()
        self.args = args

        if self.args.method == "resnet18":
            resnet = models.resnet18(pretrained=True)  
        elif self.args.method == "resnet34":
            resnet = models.resnet34(pretrained=True)
        elif self.args.method == "resnet50":
            resnet = models.resnet50(pretrained=True)

        last_layer_idx = -2
        self.resnet = nn.Sequential(*list(resnet.children())[:last_layer_idx])
        self.last_layer = nn.Sequential(*list(resnet.children())[last_layer_idx])

        self.mem = Memory(batch_size=, key_dim=512, mem_depth=FLAGS.mem_depth,num_mem_slots=FLAGS.mem_size, 
                          vocab_size=num_classes, alpha=0.5)

    def forward(self, context_images, context_labels, target_images, clear_mem=True):
        if clear_mem:

        context_sp = self.resnet(context_images).squeeze()  # 200, 2048, 7, 7  way*shot*frames
        target_sp = self.resnet(target_images).squeeze()

        context_features = self.last_layer(context_featmap)
        target_features = self.last_layer(target_features)

        dim = int(context_features.shape[1])

        context_features = context_features.reshape(-1, self.args.seq_len, dim)
        target_features = target_features.reshape(-1, self.args.seq_len, dim)

        use_sp = True
        if use_sp:
            logits = context_sp
            logits = tf.reshape(logits, [-1, FLAGS.sample_frames_per_video, 2048, 49])
            logits = tf.transpose(logits, [0, 1, 3, 2])
            num_sp_vecs = 49  # 7*7
        else:
            num_sp_vecs = 1
        
        if is_training:
            if True:
                logits = tf.reshape(logits, [-1, FLAGS.sample_frames_per_video * num_sp_vecs, 2048])
            else:
                logits = tf.reshape(logits, [-1, FLAGS.sample_frames_per_video, 2048])
        else:
            logits = tf.reshape(logits, [1, -1, 2048])