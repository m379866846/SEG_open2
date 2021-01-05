# -*- coding: utf-8 -*-
"""
Created on Fri Nov 27 10:30:59 2020

@author: newmi
"""

import pandas as pd
import jieba
import re
import torch
import numpy as np
import os
from transformers import BertModel
from transformers import BertTokenizer
from gensim.models.fasttext import load_facebook_model
from torch.nn.utils.rnn import pad_sequence


def load_data(file_path, args):
    print('Loading dataset:', file_path)
    max_size = 50
    
    df = pd.read_csv(file_path)
    y_text = df['Intent']
    categories = df['Intent'].unique()
    class_dict = {t:i for i, t in enumerate(categories)}
    y = [class_dict[label] for label in y_text]
    y = torch.tensor(y, dtype=torch.float32)
    
    if args.w2v == 'Bert':
        for index, row in df.iterrows():
            row['Sentence'] = jieba.lcut(row['Sentence'])
            row['Sentence'] = [x for x in row['Sentence'] if x != ' ']
            #print(row['Sentence'])
            newRow = ''
            former_chinese = False
            first_test = True
            
            length = 0
            for text in row['Sentence']:
                length = length + 1
                if(length == 50):
                    break
                if first_test == True:
                    first_test = False
                    newRow = newRow + text
                    if re.search(r'[\u4e00-\ufaff]', text):
                        former_chinese =  True
                else:
                    if re.search(r'[\u4e00-\ufaff]', text):
                        #Chinese
                        former_chinese = True
                        newRow = newRow + text

                    else:
                        if former_chinese == True:
                            newRow = newRow + text
                        else:
                            newRow = newRow + " " + text
                        former_chinese = False
                        
            row['Sentence'] = newRow
            df.at[index, 'Sentence'] = row['Sentence']
        x_text = df['Sentence']
        
        print('Load data finish')
        return x_text, y_text, y, class_dict

    else:
        x_text = df['Sentence']
        return x_text, y_text, y, class_dict

#W2V Data reader Helper -------------------------
def load_w2v(file_path):
    print("loading w2v")
    w2v = load_facebook_model(file_path)
    print("w2v loaded")
    return w2v

def tokenizer_w2v(query, w2v, listbuilding):
    query_token_id = []
    len_count = 0
    size_vocab, size_emb = w2v.wv.syn0.shape
    for w in query:
        len_count = len_count + 1
        if(len_count > 50):
            break
        if w in w2v.wv.vocab:
            query_token_id.append(w2v.wv.vocab[w].index)
        else:
            if listbuilding == True:
                print(w, ' not in w2v vocab, building new vocab list')
                weight = np.random.uniform(low=-0.5, high=0.5, size=(size_emb,))
                w2v.wv.add(w, weight.astype(np.float32), replace=False)
                query_token_id.append(w2v.wv.vocab[w].index)
            else:
                pass
    token_id = torch.tensor(query_token_id)
    return token_id 

def tokenize_w2v(text, w2v, listbuilding=True):
    max_len = 50
    print("Tokenizing text------------------------------------------------------------------")
    n_vocab, d_emb = w2v.wv.syn0.shape
    token_id = [tokenizer_w2v(query, w2v, listbuilding).long() for query in text]
    pad_token_id = torch.tensor([])

    counter = 0
    for sentence in token_id:
        if len(sentence) < max_len:
            pad_torch = torch.zeros((max_len - len(sentence)))
            #print(sentence.size())
            #print(pad_torch.size())
            new_sentence = torch.cat((sentence, pad_torch), -1)
        if (counter == 0):
            counter = counter + 1
            pad_token_id = torch.cat((pad_token_id, new_sentence), 0)
        else:
            if(counter == 1):
                pad_token_id = torch.stack((pad_token_id, new_sentence), 0)
                counter = counter + 1
            else:
                pad_token_id = torch.cat((pad_token_id, new_sentence.unsqueeze(0)), 0)


    #pad_token_id = pad_sequence(token_id, batch_first=True, padding_value=0)
    print("Tokenizing text end--------------------------------------------------------------")
    return pad_token_id.long(), n_vocab, d_emb
#------------------------------------------
# Bert Data reader Helper ----------------------
def tokenized_data(data_list, tokenizer, max_seq):
    tokenized_text = []
    for text in data_list:
        tokenized_text.append(tokenizer.encode(text, add_special_tokens=True)[:max_seq])
    return tokenized_text

def pad_data(token_list, max_seq):
    return torch.tensor(np.array([el + [0] * (max_seq - len(el)) for el in token_list]), dtype=torch.long)

def att_mask(input_ids):
    attention_mask = []
    for sent in input_ids:
        att_mask = [int(token_id > 0) for token_id in sent]
        attention_mask.append(att_mask)
    return attention_mask
#-----------------------------------------------
def shuffle_data(data, ifBert):
    idx_tr = torch.tensor([]).long()
    n_tr = data['n_tr']

    idx_tr = torch.randperm(n_tr, dtype=torch.long)
    if ifBert == True:
        data['x_tr_mask'] = data['x_tr_mask'][idx_tr]
        data['x_tr'] = data['x_tr'][idx_tr]
    else:
        data['x_tr'] = data['x_tr'][idx_tr]
    data['y_tr'] = data['y_tr'][idx_tr]
    return data


def read_data(file_path, args):
    key_pretrained = 'bert-base-chinese'
    max_seq = 100
    data = {}

    x_test, y_test, _, _ = load_data(file_path['path_test'], args)
    x_train, y_train, y_tr_idx, train_class_dict = load_data(file_path['path_train'], args)

    if 'path_val' in file_path.keys():
        x_val, y_val, _, _ = load_data(file_path['path_val'])

        y_val_idx = [-1 for _ in range(len(y_val))]
        y_val_outlier_idx = [-1 for _ in range(len(y_val))]
        for index, label in enumerate(y_val):
            if label in train_class_dict:
                y_val_idx[index] = train_class_dict[label]
                y_val_outlier_idx[index] = 1

        y_val_idx = torch.tensor(y_val_idx)
        y_val_outlier_idx = torch.tensor(y_val_outlier_idx)


    y_te_idx = [-1 for _ in range(len(y_test))]
    y_te_outlier_idx = [-1 for _ in range(len(y_test))]
    for index, label in enumerate(y_test):
        if label in train_class_dict:
            y_te_idx[index] = train_class_dict[label]
            y_te_outlier_idx[index] = 1


    y_tr_idx = torch.tensor(y_tr_idx, dtype=torch.int64)
    y_te_idx = torch.tensor(y_te_idx, dtype=torch.int64)
    y_te_outlier_idx = torch.tensor(y_te_outlier_idx)

    if args.w2v == 'Bert':
        print('---------------Loading bert------------')
        tokenizer = BertTokenizer.from_pretrained(file_path[key_pretrained])
        #bert_model = BertModel.from_pretrained(file_path[key_pretrained])
        print('bert loaded')


        print('--------------tokenizing data-----------')
        x_test_tokenized = tokenized_data(x_test, tokenizer, max_seq)
        x_train_tokenized = tokenized_data(x_train, tokenizer, max_seq)
        label_tokenized = tokenized_data(train_class_dict.keys(), tokenizer, max_seq)
    
        #print(label_tokenized)
        #print(x_test_tokenized)


        if 'path_val' in file_path.keys():
            x_val_tokenized = tokenized_data(x_val, tokenizer, max_seq)

        print('data tokenized')
        print('--------------padding data--------------')
        x_test_padded = pad_data(x_test_tokenized, max_seq)
        x_train_padded = pad_data(x_train_tokenized, max_seq)
        label_padded = pad_data(label_tokenized, max_seq)
        print(label_padded)


        if 'path_val' in file_path.keys():
            x_val_padded = pad_data(x_val_tokenized, max_seq)
            x_val_att = att_mask(x_val_padded)
            #print(x_val_att)
            x_val_att = torch.tensor(x_val_att)
        print('data padded')

        x_test_att = att_mask(x_test_padded)
        x_train_att = att_mask(x_train_padded)
        label_att = att_mask(label_padded)
        x_test_att = torch.tensor(x_test_att)
        x_train_att = torch.tensor(x_train_att)
        label_att = torch.tensor(label_att)


    else:
        w2v_path = args.w2v_path
        w2v = load_w2v(w2v_path)
        n_vocab, d_emb = w2v.wv.syn0.shape
        if 'path_val' in file_path.keys():
            x_val_tokenized = tokenize_w2v(x_val, w2v, listbuilding = False)
        x_test_tokenized, _, _ = tokenize_w2v(x_test, w2v, listbuilding = False)
        x_train_tokenized, data['n_vocab'], data['d_emb'] = tokenize_w2v(x_train, w2v)
        data['embedding'] = torch.from_numpy(w2v.wv.syn0)


    """
    with torch.no_grad():
        x_train_vec = bert_model(x_train_padded)
        x_test_vec = bert_model(x_test_padded)
    """
    
    idx_tr = torch.tensor([]).long()
    idx_tr = torch.randperm(len(x_train), dtype=torch.long)
    #print(idx_tr)

    #print('x_train_att: ', x_train_att.size())

    if 'path_val' in file_path.keys():
        data['n_val'] = len(x_val)
        data['x_val'] = x_val_padded
        if args.w2v == 'Bert':
            data['x_val_mask'] = x_val_att

        data['y_val'] = y_val_idx
        data['y_val_outlier'] = y_val_outlier_idx
        data['y_val_raw'] = y_val

    data['label_padded'] = label_padded
    data['label_mask'] = label_att
    data['seen_class_dic'] = train_class_dict
    data['n_tr'] = len(x_train)
    data['n_te'] = len(x_test)
    

    if args.w2v == 'Bert':
        data['x_te_mask'] = x_test_att
        data['x_tr_mask'] = x_train_att[idx_tr]
        data['x_te'] = x_test_padded
        data['x_tr'] = x_train_padded[idx_tr]
        data['key_pretrained'] = 'bert-base-chinese'
    else:
        data['x_te'] = x_test_tokenized
        data['x_tr'] = x_train_tokenized[idx_tr]
        data['key_pretrained'] = 'FastText'

    data['y_te'] = y_te_idx
    data['y_tr'] = y_tr_idx[idx_tr]
    data['y_te_outlier'] = y_te_outlier_idx
    data['y_te_raw'] = y_test



    

    data['n_seen_class'] = len(train_class_dict.keys())

    return data



