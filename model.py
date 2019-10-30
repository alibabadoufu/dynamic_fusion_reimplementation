##########################
# Implementation of Dynamic Fusion with Intra- and Inter-modality Attention Flow for Visual Question Answering (DFAF)
# Paper Link: https://arxiv.org/abs/1812.05252
# Code Author: Kaihua Tang
# Modified by Jasper Lai Woen Yon
##########################

import math
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch.autograd import Variable
from torch.nn.utils import weight_norm
from torch.nn.utils.rnn import pack_padded_sequence

import config
import word_embedding


class Net(nn.Module):
    """
        Implementation of Dynamic Fusion with Intra- and Inter-modality Attention Flow for Visual Question Answering (DFAF)
        Based on code from https://github.com/Cyanogenoid/vqa-counting
    """
    def __init__(self, words_list):
        super(Net, self).__init__()
        self.question_features      = config.question_features
        self.vision_features        = config.output_features
        self.spatial_features       = config.spatial_features
        self.hidden_features        = config.hidden_features
        self.num_inter_head         = config.num_inter_head
        self.num_intra_head         = config.num_intra_head
        self.num_block              = config.num_block
        self.spa_block              = config.spa_block
        self.que_block              = config.que_block
        self.visual_normalization   = config.visual_normalization

        self.iteration              = config.iteration

        assert(self.hidden_features % self.num_inter_head == 0)
        assert(self.hidden_features % self.num_intra_head == 0)
        words_list = list(words_list)
        words_list.insert(0, '__unknown__')  #  add 'unk' key to the available vocab list

        self.text = word_embedding.TextProcessor(
            classes             = words_list,
            embedding_features  = 300,
            lstm_features       = self.question_features,
            drop                = 0.1,
        )

        self.interIntraBlocks = SingleBlock(
            num_block       = self.num_block,           # 2
            spa_block       = self.spa_block,           # 2
            que_block       = self.que_block,           # 2
            iteration       = self.iteration,           # 2
            v_size          = self.vision_features,     # 2048
            q_size          = self.question_features,   # 1280
            b_size          = self.spatial_features,    # 4
            output_size     = self.hidden_features,     # 512
            num_inter_head  = self.num_inter_head,      # 8
            num_intra_head  = self.num_intra_head,      # 8
            drop            = 0.1,
        )

        self.classifier = Classifier(
            in_features  = self.hidden_features,
            mid_features = config.mid_features,
            out_features = config.max_answers,
            drop         = config.classifier_dropout)

        for m in self.modules():
            if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
                init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, v, b, q, v_mask, q_mask):
        '''
        v: visual feature       [batch, 2048, num_obj]
        b: bounding box         [batch, 4,    num_obj]
        q: question             [batch, max_q_len]
        v_mask: number of obj   [batch, max_obj]   1 is obj,  0 is none
        q_mask: question length [batch, max_len]   1 is word, 0 is none
        answer: predict logits  [batch, config.max_answers]
        '''
        # prepare v & q features
        v = v.transpose(1,2).contiguous()
        b = b.transpose(1,2).contiguous()
        q = self.text(q)  # [batch, max_len, 1280]
        if self.visual_normalization:
            v = v / (v.norm(p=2, dim=2, keepdim=True) + 1e-12).expand_as(v) # [batch, max_obj, 2048]
        v_mask = v_mask.float()
        q_mask = q_mask.float()
        
        v, q, b = self.interIntraBlocks(v, q, b, v_mask, q_mask)
        answer  = self.classifier(v, b, q, v_mask, q_mask)

        return answer


class Fusion(nn.Module):
    """ Crazy multi-modal fusion: negative squared difference minus relu'd sum
    """
    def __init__(self):
        super().__init__()

    def forward(self, x, y):
        # found through grad student descent ;)
        return - (x - y)**2 + F.relu(x + y)


class ReshapeBatchNorm(nn.Module):
    def __init__(self, feat_size, affine=True):
        super(ReshapeBatchNorm, self).__init__()
        self.bn = nn.BatchNorm1d(feat_size, affine=affine)

    def forward(self, x):
        assert(len(x.shape) == 3)
        batch_size, num, _ = x.shape
        x = x.view(batch_size * num, -1)
        x = self.bn(x)
        return x.view(batch_size, num, -1)


class Classifier(nn.Sequential):
    def __init__(self, in_features, mid_features, out_features, drop=0.0):
        super(Classifier, self).__init__()
        self.drop = nn.Dropout(drop)
        self.relu = nn.ReLU()
        #self.fusion = Fusion()
        self.lin1 = nn.Linear(in_features, mid_features)
        self.lin2 = nn.Linear(mid_features, out_features)
        self.bn = nn.BatchNorm1d(mid_features)

    def forward(self, v, b, q, v_mask, q_mask):
        """
        v: visual feature      [batch, num_obj, 512]
        q: question            [batch, max_len, 512]
        v_mask                 [batch, num_obj]
        q_mask                 [batch, max_len]
        """
        v_mean = (v * v_mask.unsqueeze(2)).sum(1) / v_mask.sum(1).unsqueeze(1)
        q_mean = (q * q_mask.unsqueeze(2)).sum(1) / q_mask.sum(1).unsqueeze(1)
        
        out = self.lin1(self.drop(v_mean * q_mean))
        out = self.lin2(self.drop(self.relu(self.bn(out))))
        return out


class SingleBlock(nn.Module):
    """
    Single Block Inter-/Intra-modality stack multiple times
    """
    def __init__(self, num_block, spa_block, que_block, iteration, v_size, b_size, q_size, output_size, num_inter_head, num_intra_head, drop=0.0):
        super(SingleBlock, self).__init__()
        self.v_size         = v_size             # 2048
        self.q_size         = q_size             # 1028
        self.b_size         = b_size             # 4
        self.output_size    = output_size        # 512
        self.num_inter_head = num_inter_head     # 8
        self.num_intra_head = num_intra_head     # 8
        self.num_block      = num_block          # 2
        self.spa_block      = spa_block          # 2
        self.que_block      = que_block          # 2

        self.iteration      = iteration

        self.v_lin = nn.Linear(v_size, output_size)
        self.b_lin = nn.Linear(b_size, output_size)
        self.q_lin = nn.Linear(q_size, output_size)

        self.interBlock       = InterModalityUpdate(output_size, output_size, output_size, num_inter_head, drop)
        self.intraBlock       = DyIntraModalityUpdate(output_size, output_size, output_size, num_intra_head, drop)

        if config.exp_id == 2:
            self.n_relations    = 8
            self.dim_g          = int(output_size / self.n_relations)

            self.RelationModule   = RelationModule(n_relations      = self.n_relations, 
                                                hidden_dim       = output_size, 
                                                key_feature_dim  = self.dim_g, 
                                                geo_feature_dim  = self.dim_g,
                                                drop             = drop)

        self.drop = nn.Dropout(drop)

    def forward(self, v, q, b, v_mask, q_mask):
        """
            v: visual feature      [batch, num_obj, feat_size]
            q: question            [batch, max_len, feat_size]
            v_mask                 [batch, num_obj]
            q_mask                 [batch, max_len]
        """
        # transfor features
        v = self.v_lin(self.drop(v))
        b = self.b_lin(self.drop(b))
        q = self.q_lin(self.drop(q))

        v_init = v.clone()
        b_init = b.clone()
        q_init = q.clone()

        for i in range(self.num_block):
            v, q = self.interBlock(v, q, v_mask, q_mask)
            v, q = self.intraBlock(v, q, v_mask, q_mask)

            if config.exp_id == 2:
                v = self.RelationModule(v, b, v_mask, q, q_mask)

        return v,q,b


class RelationModule(nn.Module):
    def __init__(self, n_relations=8, hidden_dim=512, key_feature_dim=64, geo_feature_dim=64, drop=0.0):
        super(RelationModule, self).__init__()

        self.Nr        = n_relations
        self.dim_g     = geo_feature_dim
        self.relation  = nn.ModuleList()

        self.relu      = nn.ReLU(inplace=True)
        self.drop      = nn.Dropout(drop)
        self.sigmoid   = nn.Sigmoid()

        # question
        self.q_compress = nn.Linear(hidden_dim, 1)
        self.q_map_obj  = nn.Linear(hidden_dim, config.output_size**2)

        # for N in range(self.Nr):
        self.relation = RelationUnit(n_relations, hidden_dim, key_feature_dim, geo_feature_dim)

    def forward(self, v, b, v_mask, q, q_mask):

        # v: torch.Size([bs, num_obj, 512])
        # q: torch.Size([bs, q_len, 512])
        # b: torch.Size([bs, num_obj, 512])

        q = q * q_mask.unsqueeze(2)

        # summarize question
        q_compress = self.q_compress(q)
        q_mask     = (q_mask == 0).unsqueeze(-1).expand_as(q_compress)      # (bs, q_len, 1)
        q_compress = q_compress.masked_fill_(q_mask, -float('inf'))         # (bs, q_len, 1)
        q_sscore   = self.sigmoid(q_compress)                               # (bs, q_len, 1)
        q_summary  = torch.bmm(q.transpose(1,2), q_sscore).squeeze(-1)      # (bs, 512)
        q_map_obj  = self.relu(self.q_map_obj(q_summary))                   # (bs, 36*36)

        concat = self.relation(v, b, q_map_obj, v_mask)

        return concat + v

class RelationUnit(nn.Module):
    def __init__(self, n_relations=8, hidden_dim=512, key_feature_dim=64, geo_feature_dim=64):
        super(RelationUnit, self).__init__()

        self.n_relations = n_relations

        self.dim_g  = geo_feature_dim
        self.dim_k  = key_feature_dim

        self.WG_1   = nn.Linear(hidden_dim, hidden_dim)
        self.WG_2   = nn.Linear(hidden_dim, hidden_dim)

        self.WK     = nn.Linear(hidden_dim, hidden_dim, bias=True)
        self.WQ     = nn.Linear(hidden_dim, hidden_dim, bias=True)
        self.WV     = nn.Linear(hidden_dim, hidden_dim, bias=True)

        self.relu   = nn.ReLU(inplace=True)
        self.softmax = nn.Softmax()

    def forward(self, v, b, q_map_obj, v_mask):

        # v: torch.Size([256, 36, 512])
        # b: torch.Size([256, 36, 512])

        bs, num_obj, dim  = v.size()
        q_map_obj         = q_map_obj.view(bs, num_obj, -1).unsqueeze(1)

        dim_per_rel = dim // self.n_relations
        mask_reshape = (bs, 1, 1, num_obj)

        def shape(x)  : return x.view(bs, -1, self.n_relations, dim_per_rel).transpose(1, 2)
        def unshape(x): return x.transpose(1, 2).contiguous().view(bs, -1, self.n_relations * dim_per_rel)

        w_g_1 = self.WG_1(b) * v_mask.unsqueeze(2)                      # (bs, n_rel, num_obj, 64)
        w_g_2 = self.WG_2(b) * v_mask.unsqueeze(2)                      # (bs, n_rel, num_obj, 64)

        w_g_1 = shape(w_g_1)
        w_g_2 = shape(w_g_2)

        w_g = self.relu(torch.matmul(w_g_1, w_g_2.transpose(2,3)))      # (bs, n_rel, num_obj, num_obj)
        w_g = w_g + q_map_obj                                           # (bs, n_rel, num_obj, num_obj)

        w_k = shape(self.WK(v))                                         # (bs, n_rel, num_obj, 64)
        w_k = w_k.view(bs, -1, num_obj, 1, self.dim_k)                  # (bs, n_rel, num_obj, 1, 64)

        w_q = shape(self.WQ(v))                                         # (bs, n_rel, num_obj, 64)
        w_q = w_q.view(bs, -1, 1, num_obj, self.dim_k)                  # (bs, n_rel, 1, num_obj, 64)

        scaled_dot = torch.sum((w_k * w_q), dim=-1)                     # (bs, n_rel, num_obj, num_obj)
        scaled_dot = scaled_dot / math.sqrt(self.dim_k)                 # (bs, n_rel, num_obj, num_obj)

        w_mn    = torch.log(torch.clamp(w_g, min = 1e-6)) + scaled_dot  # (bs, n_rel, num_obj, num_obj)
        v_mask  = (v_mask == 0).view(mask_reshape).expand_as(w_mn)      # (bs, n_rel, num_obj, num_obj)
        w_mn    = w_mn.masked_fill_(v_mask, -float('inf'))              # (bs, n_rel, num_obj, num_obj)
        w_mn    = F.softmax(w_mn, dim=-1)                               # (bs, n_rel, num_obj, num_obj)

        w_v = shape(self.WV(v))                     # (bs, n_rel, num_obj, 64)

        output = torch.matmul(w_mn, w_v)     # (bs, n_rel, num_obj, 64)
        output = unshape(output)

        return output

class InterModalityUpdate(nn.Module):
    """
        Inter-modality Attention Flow
    """
    def __init__(self, v_size, q_size, output_size, num_head, drop=0.0):

        super(InterModalityUpdate, self).__init__()
        self.v_size      = v_size        # 512
        self.q_size      = q_size        # 512
        self.output_size = output_size   # 512
        self.num_head    = num_head      # 8

        self.v_lin = nn.Linear(v_size, output_size * 3)
        self.q_lin = nn.Linear(q_size, output_size * 3)

        self.v_output = nn.Linear(output_size + v_size, output_size)
        self.q_output = nn.Linear(output_size + q_size, output_size)

        self.relu = nn.ReLU()
        self.drop = nn.Dropout(drop)

    def forward(self, v, q, v_mask, q_mask):
        """
            v: visual feature      [batch, num_obj, feat_size]
            q: question            [batch, max_len, feat_size]
            v_mask                 [batch, num_obj]
            q_mask                 [batch, max_len]
        """

        batch_size, num_obj = v_mask.shape
        _         , max_len = q_mask.shape

        dim_per_head = self.output_size // self.num_head

        def shape(x)    : return x.view(batch_size, -1, self.num_head, dim_per_head).transpose(1, 2)
        def unshape(x)  : return x.transpose(1, 2).contiguous().view(batch_size, -1, self.num_head*dim_per_head)

        v = v * v_mask.unsqueeze(2)
        q = q * q_mask.unsqueeze(2)

        # transfor features
        v_trans = self.v_lin(self.drop(self.relu(v)))
        q_trans = self.q_lin(self.drop(self.relu(q)))

        # split for different use of purpose
        v_k, v_q, v_v = torch.split(v_trans, v_trans.size(2) // 3, dim=2)
        q_k, q_q, q_v = torch.split(q_trans, q_trans.size(2) // 3, dim=2)

        # transform features
        v_k = shape(v_k)  # (batch_size, num_head, num_obj, dim_per_head)
        v_q = shape(v_q)  # (batch_size, num_head, num_obj, dim_per_head)
        v_v = shape(v_v)  # (batch_size, num_head, num_obj, dim_per_head)

        q_k = shape(q_k)  # (batch_size, num_head, max_len, dim_per_head)
        q_q = shape(q_q)  # (batch_size, num_head, max_len, dim_per_head)
        q_v = shape(q_v)  # (batch_size, num_head, max_len, dim_per_head)

        # inner product
        q2v = torch.matmul(v_q, q_k.transpose(2,3)) / math.sqrt(dim_per_head)   # (batch_size, num_head, num_obj, max_len)
        v2q = torch.matmul(q_q, v_k.transpose(2,3)) / math.sqrt(dim_per_head)   # (batch_size, num_head, max_len, num_obj)

        q_mask = (q_mask == 0).unsqueeze(1).unsqueeze(1).expand_as(q2v)
        v_mask = (v_mask == 0).unsqueeze(1).unsqueeze(1).expand_as(v2q)

        # set padding object/word attention to negative infinity & normalized by square root of hidden dimension
        q2v = q2v.masked_fill(q_mask, -float('inf'))   # (batch_size, num_head, num_obj, max_len)
        v2q = v2q.masked_fill(v_mask, -float('inf'))   # (batch_size, num_head, max_len, num_obj)

        # softmax attention
        interMAF_q2v = F.softmax(q2v.float(), dim=-1).type_as(q2v) # (batch_size, num_head, num_obj, max_len) over max_len
        interMAF_v2q = F.softmax(v2q.float(), dim=-1).type_as(v2q) # (batch_size, num_head, max_len, num_obj) over num_obj

        v_update = unshape(torch.matmul(interMAF_q2v, q_v))
        q_update = unshape(torch.matmul(interMAF_v2q, v_v))

        # update new feature
        cat_v = torch.cat((v, v_update), dim=2)
        cat_q = torch.cat((q, q_update), dim=2)

        updated_v = self.v_output(self.drop(cat_v))
        
        if config.exp_id == 2:
            updated_q = q
        else:
            updated_q = self.q_output(self.drop(cat_q))

        return updated_v, updated_q

class DyIntraModalityUpdate(nn.Module):
    """
    Dynamic Intra-modality Attention Flow
    """
    def __init__(self, v_size, q_size, output_size, num_head, drop=0.0):

        super(DyIntraModalityUpdate, self).__init__()
        self.v_size = v_size
        self.q_size = q_size
        self.output_size = output_size
        self.num_head = num_head

        self.v4q_gate_lin = nn.Linear(v_size, output_size)
        self.q4v_gate_lin = nn.Linear(q_size, output_size)

        self.v_lin = nn.Linear(v_size, output_size * 3)
        self.q_lin = nn.Linear(q_size, output_size * 3)

        self.v_output = nn.Linear(output_size, output_size)
        self.q_output = nn.Linear(output_size, output_size)

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.drop = nn.Dropout(drop)
    
    def forward(self, v, q, v_mask, q_mask):

        """
        v: visual feature      [batch, num_obj, feat_size]
        q: question            [batch, max_len, feat_size]
        v_mask                 [batch, num_obj]
        q_mask                 [batch, max_len]
        """
        batch_size, num_obj = v_mask.shape
        _         , max_len = q_mask.shape

        dim_per_head = self.output_size // self.num_head

        # average pooling
        v_mean      = (v * v_mask.unsqueeze(2)).sum(1) / v_mask.sum(1).unsqueeze(1)
        q_mean      = (q * q_mask.unsqueeze(2)).sum(1) / q_mask.sum(1).unsqueeze(1)

        # conditioned gating vector
        v4q_gate    = self.sigmoid(self.v4q_gate_lin(self.drop(self.relu(v_mean)))).unsqueeze(1) #[batch, 1, feat_size]
        q4v_gate    = self.sigmoid(self.q4v_gate_lin(self.drop(self.relu(q_mean)))).unsqueeze(1) #[batch, 1, feat_size]

        # key, query, value
        v_trans = self.v_lin(self.drop(self.relu(v)))
        q_trans = self.q_lin(self.drop(self.relu(q)))

        # mask all padding object/word features
        v_trans = v_trans * v_mask.unsqueeze(2)
        q_trans = q_trans * q_mask.unsqueeze(2)

        # split for different use of purpose
        v_k, v_q, v_v = torch.split(v_trans, v_trans.size(2) // 3, dim=2)
        q_k, q_q, q_v = torch.split(q_trans, q_trans.size(2) // 3, dim=2)

        def shape_gate(x)   : return x.view(batch_size, self.num_head, dim_per_head).unsqueeze(2)
        def shape(x)        : return x.view(batch_size, -1, self.num_head, dim_per_head).transpose(1, 2)
        def unshape(x)      : return x.transpose(1, 2).contiguous().view(batch_size, -1, self.num_head*dim_per_head)

        # transform features
        v_k = shape(v_k)  # (batch_size, num_head, num_obj, dim_per_head)
        v_q = shape(v_q)  # (batch_size, num_head, num_obj, dim_per_head)
        v_v = shape(v_v)  # (batch_size, num_head, num_obj, dim_per_head)

        q_k = shape(q_k)  # (batch_size, num_head, max_len, dim_per_head)
        q_q = shape(q_q)  # (batch_size, num_head, max_len, dim_per_head)
        q_v = shape(q_v)  # (batch_size, num_head, max_len, dim_per_head)

        # apply conditioned gate
        new_vq = (1 + shape_gate(q4v_gate)) * v_q   # (batch_size, nhead, num_obj, dim_per_head)
        new_vk = (1 + shape_gate(q4v_gate)) * v_k   # (batch_size, nhead, num_obj, dim_per_head)
        new_qq = (1 + shape_gate(v4q_gate)) * q_q   # (batch_size, nhead, max_len, dim_per_head)
        new_qk = (1 + shape_gate(v4q_gate)) * q_k   # (batch_size, nhead, max_len, dim_per_head)

        # multi-head attention
        v2v = torch.matmul(new_vq, new_vk.transpose(2,3)) / math.sqrt(dim_per_head)
        q2q = torch.matmul(new_qq, new_qk.transpose(2,3)) / math.sqrt(dim_per_head)

        # masking
        v_mask = (v_mask == 0).unsqueeze(1).unsqueeze(1).expand_as(v2v)
        q_mask = (q_mask == 0).unsqueeze(1).unsqueeze(1).expand_as(q2q)

        # set padding object/word attention to negative infinity & normalized by square root of hidden dimension
        v2v = v2v.masked_fill(v_mask, -float('inf'))   # (batch_size, num_head, num_obj, max_len)
        q2q = q2q.masked_fill(q_mask, -float('inf'))   # (batch_size, num_head, max_len, num_obj)

        # attention score
        dyIntraMAF_v2v = F.softmax(v2v, dim=-1).type_as(v2v)
        dyIntraMAF_q2q = F.softmax(q2q, dim=-1).type_as(q2q)

        v_update = unshape(torch.matmul(dyIntraMAF_v2v, v_v))
        q_update = unshape(torch.matmul(dyIntraMAF_q2q, q_v))

        # update
        updated_v = self.v_output(self.drop(v + v_update))
        updated_q = self.q_output(self.drop(q + q_update))

        return updated_v, updated_q