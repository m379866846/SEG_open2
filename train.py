# -*- coding: utf-8 -*-
"""
Created on Fri Nov 27 10:33:12 2020

@author: newmi
"""

from visualize import *
from data_reader import *
import transformers
import model_encoder as model_out
import model_classifier
from sklearn.metrics import roc_auc_score
import os
import time
import torch
import numpy as np
from sklearn.covariance import EllipticEnvelope
from sklearn import svm
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from torch.nn import functional as F
import torch.nn as nn
from torch.autograd import Variable
import torch.utils.data as data

def encode_onehot(labels, n_classes):
    onehot = torch.FloatTensor(labels.size()[0], n_classes)
    labels = labels.data
    if labels.is_cuda:
        onehot = onehot.cuda()
    onehot.zero_()
    onehot.scatter_(1, labels.view(-1, 1), 1)
    return onehot


def train(data, config):
    config_outlier = config['outlier']
    device = config['device']
    print(device)
    batch_size = config['batch_size']
    n_tr = data['n_tr']
    config['lowest_lost'] = 999
    n_epoch = config_outlier['n_epoch']
    config['test_loss'] = []
    config['test_acc'] = []
    config['scl_loss'] = []
    config['test_auc'] = []
    class_token_ids = data['label_padded'].to(device)
    s2v = True if config['s2v'] == 's2v' else False
    in_dim = 768 if s2v == True else config['lstm_size'] * 2
    config['patient_stop'] = 0
    config['patient_reduce'] = 0
    bert_type = config['bert_type']
    w2v = config['w2v']

    if (s2v == False):
        if(w2v == 'FastTest'):
            encoder = model_out.FastTextLayer(data['embedding']).to(device)
        else:
            if (bert_type == 'DistilBert'):
                encoder = model_out.Transformer_disbert(config['bert_path'], s2v=s2v).to(device)
            else:
                encoder = model_out.Transformer_bert(config['bert_path'], s2v=s2v).to(device)
        feat_extractor = model_out.lstm_feat_extract(config['word_emb_size'], config['lstm_size'], config['att_dim'], config['feat_dim']).to(device)

    else:
        if(config['bert_type'] == 'DistilBert'):
            encoder = model_out.Transformer_disbert(config['bert_path'], s2v=s2v).to(device) 
        else:
            encoder = model_out.Transformer_bert(config['bert_path'], s2v=s2v).to(device)
        feat_extractor = model_out.Reduction(config['word_emb_size'], config['feat_dim']).to(device)   

    print('number of seen class', data['n_seen_class'])

    if (config['classifier'] == 'LMCL'):
        model_predictor = model_classifier.LMCL(data['n_seen_class'], config['feat_dim'], scl=config['scl'], t_scale=config['scl_temp'], lam2=config['scl_lam']).to(device)
    elif (config['classifier'] == 'LGML'):
        model_predictor = model_classifier.LGMLoss(data['n_seen_class'], config['feat_dim'], alpha=config['lgml_alpha'], scl=config['scl'], t_scale=config['scl_temp'], lam2=config['scl_lam']).to(device)
    elif (config['classifier'] == 'LSoftmax'):
        model_predictor = model_classifier.LSoftmax(data['n_seen_class'], config['feat_dim'], scl=config['scl'], t_scale=config['scl_temp'], lam2=config['scl_lam']).to(device)
    
    
    optimizer_classifier = torch.optim.Adam(list(model_predictor.parameters()), lr=config_outlier['lr'])
    optimizer_encoder = torch.optim.Adam(list(encoder.parameters()), lr = config_outlier['bert_lr'])
    optimizer_feat_extractor = torch.optim.Adam(list(feat_extractor.parameters()), lr=config_outlier['lr'])

    time_avg = 0.0
    n_batch = (n_tr - 1)//batch_size + 1
    for i_epoch in range(n_epoch):
        if(config['patient_reduce'] == 3):
            print('reducing learning rate')
            config['patient_reduce'] = 0
            config_outlier['bert_lr'] = config_outlier['bert_lr']/2
            config_outlier['lr'] = config_outlier['lr']/2
            print('bert lr:', config_outlier['bert_lr'])
            print('lr:', config_outlier['lr'])
            optimizer_classifier = torch.optim.Adam(list(model_predictor.parameters()), lr=config_outlier['lr'])
            optimizer_encoder = torch.optim.Adam(list(encoder.parameters()), lr = config_outlier['bert_lr'])
            optimizer_feat_extractor = torch.optim.Adam(list(feat_extractor.parameters()), lr=config_outlier['lr'])
        
        if (config['patient_stop'] == 9):
            print('stop training as patient stop reach 9')
            #test_outlier_detection(data['x_te'], data['y_te'], data['y_te_outlier'], encoder, model_predictor, batch_size, device, y_te_raw=data['y_te_raw'], seen_class_dict=data['seen_class_dic'], prefix=save_path, config=config, x_te_mask=data['x_te_mask'], algorithm=anom_algo, feat_extractor=feat_extractor, save=True)
            break

        if (config['w2v'] == 'Bert'):
            data = shuffle_data(data, True)
        else:
            data = shuffle_data(data, True)

        encoder.train()
        model_predictor.train()
        feat_extractor.train()
        
        acc_avg = 0.0
        loss_avg = 0.0
        correct = 0
        total = 0


        time_start = time.time()
        for i_batch in range(n_batch):
            index = range(i_batch * batch_size, min((i_batch + 1) * batch_size, n_tr))
            batch_x = data['x_tr'][index, :].to(device)

            if (config['w2v'] == 'Bert'):
                batch_x_mask = data['x_tr_mask'][index, :].to(device)
                label_mask = data['label_mask'].to(device)
            else:
                batch_x_mask = None
                label_mask = None
                
            batch_y = data['y_tr'][index].to(device)
            if batch_x.shape[0] == 0:
                pass
            
            if (config['w2v'] == 'Bert'):
                batch_tr_emb = feat_extractor(encoder(batch_x, batch_x_mask))
                class_emb = feat_extractor(encoder(class_token_ids, label_mask)) if config['outlier']['use_labels'] else None

            else:
                batch_tr_emb = feat_extractor(encoder(batch_x))
                class_emb = feat_extractor(encoder(class_token_ids)) if config['outlier']['use_labels'] else None


            batch_logits = model_predictor(batch_tr_emb, labels=batch_y.long(), device=device, class_emb=class_emb)


            loss = model_predictor.loss
            optimizer_encoder.zero_grad()
            optimizer_feat_extractor.zero_grad()     
            optimizer_classifier.zero_grad()
            loss.backward()
            optimizer_encoder.step()
            optimizer_feat_extractor.step()
            optimizer_classifier.step()


            batch_pred = torch.argmax(batch_logits, 1)
            correct = (batch_pred == batch_y.long()).sum().item()
            acc_avg += correct / batch_pred.shape[0]
            loss_avg += loss.item()
            time_epoch = time.time() - time_start


        acc_avg = acc_avg/n_batch
        time_avg += time_epoch
        loss_avg = loss_avg/n_batch

        print('|| epoch:%d || loss:%f ||acc_avg:%f || time_tr:%f' % (i_epoch, loss_avg, acc_avg, time_epoch))


        if (i_epoch % config['test_step'] == 0):
            if (acc_avg > 0.95):
                config['test_step'] = 1
            elif (acc_avg > 0.85):
                config['test_step'] = 3
            elif (acc_avg > 0.50):
                config['test_step'] = 5
            print('Test Performance')
            anom_algo = get_anomaly_algorithms('lof')

            if(config['w2v'] == 'Bert'):
                #if(config['s2v'] != 's2v'):
                anom_algo = fit_lof(data['x_tr'], data['y_tr'], encoder, anom_algo, config, x_tr_mask=data['x_tr_mask'], feat_extractor = feat_extractor)
                #else:
                    #anom_algo = fit_lof(data['x_tr'], data['y_tr'], encoder, anom_algo, config, x_tr_mask=data['x_tr_mask'])
            else:
                anom_algo = fit_lof(data['x_tr'], data['y_tr'], encoder, feat_extractor, anom_algo, config)

           
            save_path= config['save_path']

            class_emb = feat_extractor(encoder(class_token_ids, label_mask)) if config['outlier']['use_labels'] else None
            if(config['w2v'] == 'Bert'):
                #if(config['s2v'] != 's2v'):
                config = test_outlier_detection(data['x_te'], data['y_te'], data['y_te_outlier'], class_emb, encoder, model_predictor, batch_size, device, y_te_raw=data['y_te_raw'], seen_class_dict=data['seen_class_dic'], prefix=save_path, config=config, x_te_mask=data['x_te_mask'], algorithm=anom_algo, feat_extractor=feat_extractor)
                #else:
                    #config = test_outlier_detection(data['x_te'], data['y_te'], data['y_te_outlier'], encoder, model_predictor, batch_size, device, y_te_raw=data['y_te_raw'], seen_class_dict=data['seen_class_dic'], prefix=save_path, config=config, x_te_mask=data['x_te_mask'], algorithm=anom_algo)
            else:
                config = test_outlier_detection(data['x_te'], data['y_te'], data['y_te_outlier'], class_emb, encoder, feat_extractor, model_predictor, batch_size, device, y_te_raw=data['y_te_raw'], seen_class_dict=data['seen_class_dic'], prefix=save_path, config=config, algorithm=anom_algo)
     
    time_avg = time_avg/config_outlier['n_epoch']
    print('avg time epoch', time_epoch)

def fit_lof(x_tr, y_tr, encoder, algorithm, config, x_tr_mask=None, feat_extractor=None):
    tr_emb = []
    config_outlier = config['outlier']
    device = config['device']
    print(device)
    batch_size = config['batch_size']
    n_tr = len(x_tr)
    n_batch = (n_tr - 1)//batch_size + 1

    for i_batch in range(n_batch):
        index = range(i_batch * batch_size, min((i_batch + 1) * batch_size, n_tr))
        batch_x = x_tr[index, :].to(device)
        if (config['w2v'] == 'Bert'):
            batch_x_mask = x_tr_mask[index, :].to(device)
        else:
            batch_x_mask = None
        if batch_x.shape[0] == 0:
            pass
        if(config['w2v'] == 'Bert'):
            #if(config['s2v'] != 's2v'):
            batch_tr_emb = feat_extractor(encoder(batch_x,batch_x_mask))
            #else:
                #batch_tr_emb = encoder(batch_x, batch_x_mask)
        else:
            batch_tr_emb = feat_extractor(encoder(batch_x))
        tr_emb.append(batch_tr_emb.detach().cpu().numpy())

    tr_emb = np.concatenate(tr_emb, axis=0)
    if(config['visualize'] == True):
        cluster_visualize(tr_emb, y_tr.numpy(), 'save/training_visualization')
    algorithm.fit(tr_emb)
    return algorithm

def test_auc_score(x_te, y_te, y_te_outlier, encoder, model_predictor, model_confidence, batch_size, device,x_te_mask=None, 
                    y_te_raw=None, seen_class_dict=None, prefix=None, config=None, algorithm=None, feat_extractor=None):
    encoder.eval()
    if feat_extractor is not None:
        feat_extractor.eval()
    model_predictor.eval()
    model_confidence.eval()

    n_te = len(x_te)
    n_batch = (n_te-1)// batch_size + 1
    te_confidence = torch.tensor([]).to(device)
    with torch.no_grad():
        for i_batch in range(n_batch):
            index = range(i_batch * batch_size, min((i_batch + 1) * batch_size, n_te))
            batch_x = x_te[index,:].to(device)
            if (config['w2v'] == 'Bert'):
                batch_x_mask = x_te_mask[index,:].to(device)
            else:
                batch_x_mask = None

            if batch_x.shape[0] == 0:
                pass
            if (config['w2v'] == 'Bert'):
                #if(config['s2v'] != 's2v'):
                batch_te_emb = feat_extractor(encoder(batch_x,batch_x_mask))
                #else:
                    #batch_te_emb = encoder(batch_x, batch_x_mask)
            else:
                batch_te_emb = feat_extractor(encoder(batch_x))     

            batch_logits = model_predictor(batch_te_emb)
            batch_confidence = model_confidence(batch_te_emb)

            te_confidence = torch.cat((te_confidence, batch_confidence))
            #print(batch_logits)
    
    te_confidence_np = te_confidence.detach().cpu().clone().numpy()
    print('AUC is: ', roc_auc_score(y_te_outlier, te_confidence_np))


def test_outlier_detection(x_te, y_te, y_te_outlier, class_emb, encoder, classifier, batch_size, device,x_te_mask=None, y_te_raw=None, seen_class_dict=None, prefix=None, config=None, algorithm=None, feat_extractor=None, save=False):
    print(">> test for outlier detection <<")
    encoder.eval()
    if (feat_extractor is not None):
        feat_extractor.eval()
    classifier.eval()

    anom_algo = algorithm

    n_te = len(x_te)
    n_batch = (n_te-1)// batch_size + 1
    te_pred_class = torch.tensor([]).long().to(device)
    te_emb = torch.tensor([]).to(device)

    te_pred_class_inlier = torch.tensor([]).long().to(device)
    te_emb_inlier = torch.tensor([]).to(device)

    te_pred_class_outlier = torch.tensor([]).long().to(device)
    te_emb_outlier = torch.tensor([]).to(device)

    loss_avg = 0
    print(y_te)
    idx_outlier = np.where(y_te == -1)[0]
    idx_inlier = np.where(y_te != -1)[0]


    x_te_outlier = x_te[idx_outlier, :]
    y_te_outlier = y_te[idx_outlier]
    n_te_outlier = len(x_te_outlier)

    x_te_inlier = x_te[idx_inlier,:]
    y_te_inlier = y_te[idx_inlier]
    n_te_inlier = len(x_te_inlier)

    idx_inlier_permute = np.random.permutation([i for i in range(len(x_te_inlier))])
    x_te_inlier = x_te_inlier[idx_inlier_permute]
    y_te_inlier = y_te_inlier[idx_inlier_permute]
    idx_back_permute = [0 for i in range(len(idx_inlier_permute))]
    for i, j in enumerate(idx_inlier_permute):
        idx_back_permute[j] = i
    idx_back_permute = np.array(idx_back_permute) 
    #print(idx_inlier_permute[idx_back_permute])
    #print(idx_inlier_permute)
    #print(idx_back_permute)


    n_batch_outlier = (n_te_outlier// batch_size + 1)
    n_batch_inlier = (n_te_inlier// batch_size + 1)


    if (config['w2v'] == 'Bert'):
        x_te_mask_outlier = x_te_mask[idx_outlier,:]
        x_te_mask_inlier =x_te_mask[idx_inlier]
        x_te_mask_inlier = x_te_mask_inlier[idx_inlier_permute]

    with torch.no_grad():
        for round in range(2):
            if round == 0:
                n_batch_test = n_batch_outlier
                n_te_test = n_te_outlier
                x_te_test = x_te_outlier 
                x_te_mask_test = x_te_mask_outlier
            else:
                n_batch_test = n_batch_inlier
                n_te_test = n_te_inlier
                x_te_test = x_te_inlier
                x_te_mask_test = x_te_mask_inlier

            for i_batch in range(n_batch_test):
                index = range(i_batch * batch_size, min((i_batch + 1) * batch_size, n_te_test))
                batch_x = x_te_test[index,:].to(device)

                if batch_x.shape[0] == 0:
                    continue

                if (config['w2v'] == 'Bert'):
                    batch_x_mask = x_te_mask_test[index,:].to(device)
                    #if(config['s2v'] != 's2v'):
                    batch_te_emb = feat_extractor(encoder(batch_x, batch_x_mask))

                    
                    #else:
                        #batch_te_emb = encoder(batch_x, batch_x_mask)
                else:
                    batch_te_emb = feat_extractor(encoder(batch_x))

                if (round == 0):
                    batch_logits = classifier(batch_te_emb, class_emb=class_emb)
                    batch_pred_class = torch.max(batch_logits, dim=1)[1]

                    te_pred_class_outlier = torch.cat((te_pred_class_outlier, batch_pred_class))
                    te_emb_outlier = torch.cat((te_emb_outlier, batch_te_emb), dim=0)
                else:
                    batch_y = y_te_inlier[index].to(device)
                    #print(y_te_inlier.size())
                    classifier = classifier.to(device)
                    batch_logits = classifier(batch_te_emb, batch_y, class_emb=class_emb, device=device)
                    batch_pred_class = torch.max(batch_logits, dim=1)[1]
                    #print(batch_pred_class.size())
                    #print(batch_te_emb.size())

                    te_pred_class_inlier = torch.cat((te_pred_class_inlier, batch_pred_class))
                    #print(te_pred_class_inlier.size())
                    te_emb_inlier = torch.cat((te_emb_inlier, batch_te_emb), dim=0)
                    #print(te_emb_inlier.size())


                loss = classifier.loss
                loss_avg += loss

        idx_back_permute = torch.LongTensor(idx_back_permute).to(device)
        te_emb_inlier = te_emb_inlier[idx_back_permute, :]
        #print(idx_inlier_permute)
        te_pred_class_inlier = te_pred_class_inlier[idx_back_permute]
        #print(te_pred_class_inlier.size())
        te_emb = torch.cat((te_emb_outlier, te_emb_inlier), dim=0)
        te_pred_class = torch.cat((te_pred_class_outlier, te_pred_class_inlier))
        #print(te_emb_inlier.size())
        #print(te_pred_class_inlier.size())

    loss_avg = loss_avg/n_batch_inlier


    te_emb_np = te_emb.detach().cpu().clone().numpy()
    y_te_np = y_te.detach().cpu().clone().numpy()
    te_pred_class_np = te_pred_class.detach().cpu().clone().numpy()

    te_pred_outlier = getattr(anom_algo, 'score_samples')(te_emb_np)


    idx_inlier = torch.nonzero((y_te != -1)).squeeze(-1).numpy()
    pred_inlier = te_pred_class_np[idx_inlier]
    y_te_inlier = y_te_np[idx_inlier]
    accuracy = np.sum(pred_inlier == y_te_inlier)/len(idx_inlier)

    if (y_te_raw is not None):
        te_pred_class_list = te_pred_class_np
        new_dict = dict([(value, key) for key, value in seen_class_dict.items()])
        te_pred_class_list = [new_dict[i] for i in te_pred_class_list]

        y_outlier = [1 for _ in range(len(y_te_raw))]
        for index, label in enumerate(y_te_raw):
            if label in seen_class_dict:
                y_outlier[index] = -1

        neg_pred_outlier = np.negative(te_pred_outlier)

        print('class prediction accuracy is ', accuracy)
        #print(y_te_outlier)
        #print(neg_pred_outlier)
        current_lost = loss_avg
        print('test loss:', loss_avg)
        acc_ic = accuracy
        auc = roc_auc_score(y_outlier, neg_pred_outlier)

        config['test_loss'].append(loss_avg)
        config['test_acc'].append(accuracy)
        config['test_auc'].append(roc_auc_score(y_outlier, neg_pred_outlier))

        if (current_lost < config['lowest_lost'] or save==True):
            config['lowest_lost'] = current_lost
            print('saving prediction')
            if (config['visualize'] == True):
                cluster_visualize(te_emb_np, y_te.numpy(), 'save/testing_visualization')
            save_prediction(te_pred_outlier, te_pred_class_list, y_te_raw, config, prefix)
            config['patient_stop'] = 0
            config['patient_reduce'] = 0
        else:
            config['patient_stop'] = config['patient_stop'] + 1
            config['patient_reduce'] = config['patient_reduce'] + 1
        print('AUC is: ', roc_auc_score(y_outlier, neg_pred_outlier))



    return config

def score_to_outlier(outlier_score, threshold_min, is_soft=True):
    threshold_min = threshold_min
    outlier_score_ = np.zeros(outlier_score.shape)
    outlier_score_[outlier_score < threshold_min] = -1
    return outlier_score_

def save_prediction(outlier_score, prediction, label, config, save_path):
    d = {'outlier_score': outlier_score}
    d['prediciton'] = prediction
    d['label'] = label
    d2 = {'test_acc': config['test_acc'], 'test_auc': config['test_auc'], 'test_loss':config['test_loss']}
        #d2['scl_loss'] = config['scl_loss']
    

    df = pd.DataFrame(data=d)
    df2 = pd.DataFrame(data=d2)
    save_path_train = save_path + 'training_detail.csv'
    save_path = save_path + 'outlier_score.csv'



    df.to_csv(save_path, index=False, encoding='utf-8-sig')
    df2.to_csv(save_path_train, index=False, encoding='utf-8-sig')

def get_anomaly_algorithms(algorithm_key, outlier_fraction=0.5):

    anomaly_algorithms = dict(
        robust_covariance=EllipticEnvelope(contamination=outlier_fraction, support_fraction=0.9999),
        one_class_svm=svm.OneClassSVM(nu=outlier_fraction, kernel="rbf", gamma=0.1),
        isolation_forest=IsolationForest(behaviour='new', contamination=outlier_fraction, random_state=42),
        lof=LocalOutlierFactor(n_neighbors=20, contamination=0.05, novelty=True, n_jobs=-1),
    )
    return anomaly_algorithms[algorithm_key]