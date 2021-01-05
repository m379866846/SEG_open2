# -*- coding: utf-8 -*-
"""
Created on Fri Nov 27 10:32:40 2020

@author: newmi
"""

import torch
import time

from sklearn.covariance import EllipticEnvelope
from sklearn import svm
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor

def get_config(data, args, file_path):
    config = dict()
    config['batch_size'] = 64
    config['test_step'] = 5
    config['device'] = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config['save_path'] = file_path['save_path']
    config['bert_path'] = args.bert_path
    config['feat_dim'] = args.feat_dim
    config['att_dim'] = args.att_dim

    config['lstm_size'] = args.lstm_size
    config['word_emb_size'] = args.word_emb

    config['validation'] = args.validation

    config['w2v'] = args.w2v
    config['s2v'] = args.s2v

    config['visualize'] = args.visualize

    config['bert_type'] =args.bert_type
    config['classifier'] = args.classifier

    config['scl'] = args.scl_loss
    config['scl_lam'] = args.scl_lambda
    config['scl_temp'] = args.scl_temperature_scaling

    config['lgml_alpha'] = args.lgml_alpha

    config['outlier'] = dict(
        n_epoch = args.epoch,
        lr=1e-5,
        bert_lr = 1e-5, #Bert 3e-5
        use_labels = args.use_labels,
    )

    return config
