# -*- coding: utf-8 -*-
"""
Created on Fri Nov 27 10:32:16 2020

@author: newmi
"""


from data_reader import *
import transformers
from config import get_config
from train import *
from argparse import ArgumentParser

parser = ArgumentParser(description='config for outlier detection')
# traning related setting
parser.add_argument('-eo', '--epoch', default=200, type=int, help="number of epoch for outlier detection")
parser.add_argument('-data', '--dataset', default='SMP18', type=str, help="choose dataset in (WeLab | SMP18)")

#classifier setting
parser.add_argument('-cls', '--classifier', default='LGML', type=str, help="select classifier (LGML | LSoftmax | LMCL)")

#utility setting
parser.add_argument('-validation', '--validation', default=False, type=bool)
parser.add_argument('-visualize', '--visualize', default=False, type=bool)
parser.add_argument('-use_label', '--use_labels', default=True, type=bool)

#scl loss parameter setting
parser.add_argument('-scl', '--scl_loss', default=True, type=bool)
parser.add_argument('-scl_lam', '--scl_lambda', default=0.1, type=float)
parser.add_argument('-scl_temp', '--scl_temperature_scaling', default=0.1, type=float)

#lgml parameter setting
parser.add_argument('-lgml_alpha', '--lgml_alpha', default=0.5, type=float)

#path related setting
parser.add_argument('-res_path', '--resource_path', default='res/', type=str)
parser.add_argument('-data_path', '--data_path', default='data/', type=str)
parser.add_argument('-save_path', '--save_path', default='save/LGML-01SCL-01T/', type=str)

#Encoder-Bert & FastText Setting 
parser.add_argument('-bert_type', '--bert_type', default='DistilBert', type=str,
                    help="select classifier (DistilBert | Bert)")
parser.add_argument('-w2v', '--w2v', default='Bert', type=str,
                    help="select classifier (Bert | FastTest)")
parser.add_argument('-s2v', '--s2v', default='s2v', type=str,
                    help="select classifier (s2v | none)")


#Encoder-feat extractor setting
parser.add_argument('-lstm_size', '--lstm_size', default=128, type=int)
parser.add_argument('-fd', '--feat_dim', default=12, type=int, help="feat_dim for outlier detection")
parser.add_argument('-ad', '--att_dim', default=12, type=int, help="att_dim for outlier detection")

args = parser.parse_args()


def main(intent_input, set_input, args):
    intent_num = intent_input
    set_num = set_input
    if args.w2v == 'FastTest':
        args.word_emb = 300
    else:
        args.word_emb = 768
    resource_path = args.resource_path
    data_path = args.data_path
    save_path = args.save_path
    if args.dataset != 'WeLab':
        data_path = data_path + args.dataset + '/'
    if(args.bert_type == 'DistilBert'):
        if (args.dataset == 'SMP18'):
            bert_path = 'distilbert-base-chinese/'
        else:
            bert_path = 'distillbert-base-uncased/'
    else:
        bert_path = 'bert-base-chinese/'
    args.bert_path = bert_path

    file_path = {}
    
    if intent_num == '(L)':
        file_path['path_val'] = resource_path + data_path + + args.dataset + '_' + str(intent_num) + 'intent_validation_set' + str(set_num) + '.csv'
    file_path['path_train'] = resource_path + data_path + args.dataset + '_' + str(intent_num) + 'intent_training_set' + str(set_num) + '.csv'
    file_path['path_test'] = resource_path + data_path + args.dataset + '_' + str(intent_num) + 'intent_testing_set' + str(set_num) + '.csv'
    #file_path['path_train'] = resource_path + data_path + 'CMHK_training_set(wb).csv'
    #file_path['path_test'] = resource_path + data_path + 'CMHK_testing_set(wb).csv'

    file_path['bert-base-chinese'] = resource_path + bert_path
    #file_path['distilbert-base-chinese'] = resource + distilbert_path
    file_path['save_path'] = save_path + args.dataset + '_' + str(intent_num) + 'set' + str(set_num)
    #file_path['save_path'] = save_path + 'CMHK_' + 'set'
    file_path['w2v_model'] = 'res/w2v_model.bin'
    
    data = read_data(file_path, args)
    config = get_config(data, args, file_path)
    train(data, config)


if __name__ == '__main__':
    set_range = [i+1 for i in range(10)]
    #set_range = [9, 10]
    #set_range = [1, 2, 3]
    set_range = [1, 2, 3, 4, 5]
    intent_range = [7, 15, 23] # SMP18[7, 15, 23] || SNIP[2, 3, 5]
    for i in intent_range:
        for j in set_range:
            main(i, j, args)