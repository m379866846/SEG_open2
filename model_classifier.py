# -*- coding: utf-8 -*-

import torch.nn as nn
import torch
import numpy as np
import math
import os
import torch.nn.functional as F

def l2_normalized(feat):
    #return normalized feat
    return F.normalize(feat, p=2, dim=1)
def SCLoss(feat, labels, t):
    loss = 0
    n_sample = 0
    for i, sample1 in enumerate(feat):
        sum_log = 0
        total_temp = 0
        bottom_log = 0
        for k, sample3 in enumerate(feat):
            if (i != k):    
                bottom_log = bottom_log + torch.exp(torch.dot(sample1, sample3)/t)
        for j, sample2 in enumerate(feat):
            if (i != j and labels[i] == labels[j]):
                total_temp = total_temp + 1
                upper_log = torch.exp((torch.dot(sample1, sample2))/t)
                sum_log = sum_log + torch.log(upper_log/bottom_log)
        if(total_temp == 0):
            continue
        loss = loss - sum_log/total_temp
        n_sample = n_sample +1
    if (n_sample == 0):
        pass
    else:
        loss = loss/n_sample
    return loss

class LMCL(nn.Module):
    def __init__(self, num_classes, feat_dim, s=5, m=0.2, scl=True, t_scale=0.3, lam2_=0.1, **kwargs):
        super(LMCL, self).__init__()
        self.feat_dim = feat_dim
        self.num_classes = num_classes
        self.s = s
        self.m = m
        self.weights = nn.Parameter(torch.randn(num_classes, feat_dim))
        nn.init.kaiming_normal_(self.weights)
        self.CE = nn.CrossEntropyLoss()
        self.lam2 = lam2_
        self.scl = scl
        self.t_scale = 0.3
    def forward(self, feat, labels=None, *args, **kwargs):
        assert feat.size(1) == self.feat_dim, 'embedding size wrong'
        logits = F.linear(F.normalize(feat), F.normalize(self.weights))
        self.scl_loss = self.SCLoss(feat, labels, 0.3)

        if labels is not None:
            margin = torch.zeros_like(logits)
            index = labels.view(-1, 1).long()
            margin.scatter_(1, index, self.m)
            m_logits = self.s * (logits - margin)
            if self.scl == True:
                feat = l2_normalized(feat)
                self.scl_loss = self.SCLoss(feat, labels, self.t_scale)
                self.loss = (1 - self.lam2) * self.CE(m_logits, labels)  + self.lam2* (self.scl_loss)
            else:
                self.loss = self.CE(m_logits, labels)
        else:
            self.loss = 0
        return logits
    
class LSoftmax(nn.Module):
    def __init__(self, num_classes, feat_dim, scl=True, t_scale=0.3, lam2_=0.1, **kwargs):
        super(LSoftmax, self).__init__()
        self.fc = nn.Linear(feat_dim, num_classes)
        self.CE = nn.CrossEntropyLoss()
        self.lam2 = lam2_
        self.scl = scl
        self.t_scale = t_scale

    def forward(self, feat, labels=None, *args, **kwargs):
        logits = self.fc(feat)
        if labels is not None:
            if self.scl == True:
                feat = l2_normalized(feat)
                self.scl_loss = SCLoss(feat, labels, self.t_scale)
                self.loss = (1 - self.lam2) * self.CE(logits, labels) + self.lam2*self.scl_loss
            else:
                self.loss = self.CE(logits, labels)
                
        else:
            self.loss = 0
        return logits
    
class LGMLoss(nn.Module):
    def __init__(self, num_classes, feat_dim, alpha=0.5, lambda_=0.5, scl=True, t_scale=0.3, lam2_=0.1, **kwargs):
        super(LGMLoss, self).__init__()
        self.num_classes = num_classes
        self.alpha = alpha
        self.lambda_ = lambda_
        self.scl = scl
        self.lam2 = lam2_
        self.t_scale = t_scale
        self.means = nn.Parameter(torch.randn(num_classes, feat_dim))
        self.softmax = nn.Softmax()
        self.CE = nn.CrossEntropyLoss()
        nn.init.xavier_uniform_(self.means, gain=math.sqrt(2.0))
    
    def forward(self, feat, labels=None, device=None, class_emb=None, *args, **kwargs):
        #print(feat.size())
        if self.scl == True:
            feat = l2_normalized(feat)
        batch_size = feat.size()[0]
        XY = torch.matmul(feat, torch.transpose(self.means, 0, 1))
        XX = torch.sum(feat ** 2, dim=1, keepdim=True)
        YY = torch.sum(torch.transpose(self.means, 0, 1)**2, dim=0, keepdim=True)
        neg_sqr_dist = -0.5 * (XX - 2.0 * XY + YY)

        if labels is not None:
            #centroid update with intent labels
            #print('Not None' if class_emb is not None else None)

            means = self.means if class_emb is None else class_emb
            if self.scl == True:
                means = l2_normalized(means)
            labels_reshped = labels.view(labels.size()[0], -1)  # [bsz] -> [bsz, 1]
            ALPHA = torch.zeros(batch_size, self.num_classes).to(device)
            ALPHA = ALPHA.scatter_(1, labels_reshped,self.alpha)
            K = ALPHA + torch.ones([batch_size, self.num_classes]).to(device)
            logits_with_margin = torch.mul(neg_sqr_dist, K)
            means_batch = torch.index_select(means, dim=0, index=labels)
            loss_margin = (torch.sum((feat - means_batch)**2) / 2) * (1. / batch_size)
            if self.scl == True:
                self.scl_loss = SCLoss(feat, labels, self.t_scale)
                self.loss = (1-self.lam2) * self.CE(logits_with_margin, labels) + self.lam2*self.scl_loss + self.lambda_*loss_margin
            else:
                self.loss = self.CE(logits_with_margin, labels) + self.lambda_*loss_margin
        else:
            self.loss = 0
        return neg_sqr_dist

    