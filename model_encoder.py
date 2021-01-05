# -*- coding: utf-8 -*-
"""
Created on Fri Nov 27 10:31:46 2020

@author: newmi
"""
import torch.nn as nn
import torch
from transformers import BertModel, BertForSequenceClassification
#from transformers.modeling_distilbert import DistilBertModel, DistilBertPreTrainedModel
from transformers import DistilBertModel
import numpy as np
import math


class Transformer_bert(nn.Module):
    def __init__(self, bert_path, s2v=False):
        super(Transformer_bert, self).__init__()
        print('loading bert')
        self.encoder = BertModel.from_pretrained('res/' + bert_path)
        self.s2v = s2v
    def forward(self, x, x_mask):
        if self.s2v == True:
            emb = self.encoder(x, attention_mask=x_mask)[1] #[bst, 768]
        else:
            emb = self.encoder(x, attention_mask=x_mask)[0] #[bst, max_seq, 768] 
        return emb #[bst, max_seq, 768]

class Transformer_disbert(nn.Module):
    def __init__(self, bert_path, s2v=False):
        super().__init__()
        self.encoder = DistilBertModel.from_pretrained('res/' + bert_path)
        self.s2v = s2v
    def forward(self, x, x_mask):
        if(self.s2v == True):
            return self.encoder(x, attention_mask=x_mask)[0][:, 0] #[bst, 768]
        else:
            return self.encoder(x, attention_mask=x_mask)[0] #[bst, max_seq, 768]  

class Reduction(nn.Module):
    def __init__(self, in_dim, feat_dim, dropout=0.7):
        super(Reduction, self).__init__()
        self.reduction = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_dim, in_dim),
            nn.ReLU(),
            nn.Linear(in_dim, feat_dim),
        )            
    def forward(self, emb):
        emb = self.reduction(emb)
        return emb


class FastTextLayer(nn.Module):
    def __init__(self, embedding, dropout=0.8):
        super(FastTextLayer, self).__init__()
        self.embed = nn.Embedding.from_pretrained(embedding.clone(), freeze=False, padding_idx=0)
    def forward(self, x):
        emb = self.embed(x) #[bsz, max_len, dim_emb]
        return emb


class SelfAttLayer(nn.Module):
    def __init__(self, in_dim, emb_att):
        super(SelfAttLayer, self).__init__()
        self.layer_att = nn.Sequential(
            #nn.Dropout(0.8),
            nn.Linear(in_dim, emb_att),
            nn.Tanh(),
            nn.Linear(emb_att, 1),
            nn.Softmax(dim=-1),
        )
    def forward(self, emb_ctx):  # [bsz, T, 2*d_ctx]
        attention = self.layer_att(emb_ctx)  # [bsz, T, 1]
        #print(attention.size())
        emb_ctx = emb_ctx.permute(0, 2, 1)  
        emb_aggregate = torch.matmul(emb_ctx, attention)
        emb_aggregate = torch.squeeze(emb_aggregate, -1)
        return emb_aggregate #[bst, d_ctx]


class lstm_feat_extract(nn.Module):
    def __init__(self, in_dim, out_dim, emb_att, feat_dim ,n_layer=2, dropout=0.5):
        super(lstm_feat_extract, self).__init__()
        self.encoder = nn.LSTM(in_dim, out_dim, n_layer, dropout=dropout, bidirectional=True, batch_first=True) #[bst, max_seq, 2*out_dim]
        self.self_att = SelfAttLayer(2 * out_dim, emb_att) #[bst, 2*out_dim]
        self.reduction = Reduction(2 * out_dim, feat_dim)
    def forward(self, x):
        out_logits, _ = self.encoder(x)
        emb_out = self.self_att(out_logits) 
        emb_out = self.reduction(emb_out)
        #emb_out = self.sigmoid(emb_out)
        return emb_out #[bst, 2*out_dim]
    
    
    
    