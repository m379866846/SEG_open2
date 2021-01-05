# -*- coding: utf-8 -*-
"""
Created on Fri Nov 27 10:33:23 2020

@author: newmi
"""

import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import random


def cluster_visualize(sentence_embeddings_np, labels, savepath):

    X_embedded = TSNE(n_components=3).fit_transform(sentence_embeddings_np)
    intent_num = len(list(set(labels)))
    new_data_set = dict()

    new_data_set['Embed_x'] = X_embedded[:, 0]
    new_data_set['Embed_y'] = X_embedded[:, 1]
    new_data_set['Embed_z'] = X_embedded[:, 2]
    scatter_x = new_data_set['Embed_x']
    scatter_y = new_data_set['Embed_y']
    scatter_z = new_data_set['Embed_z']

    # colors
    colors = [[round(random.uniform(0.0,1.0), 1), round(random.uniform(0.0,1.0), 1), round(random.uniform(0.0,1.0), 1)] for i in range(intent_num)]
    #colors = colors.reshape(1,-1)

    group = labels
    cdict = {label:colors[i] for i, label in enumerate(list(set(labels)))}

    
    # plot graph
    fig = plt.figure(figsize=(16,9))
    ax = fig.gca(projection='3d')
    for g in np.unique(group):
        ix = np.where(group == g)
        if (g == -1):
            ax.scatter(scatter_x[ix], scatter_y[ix], scatter_z[ix], c=cdict[g], label = g, s = 5, alpha = 0)
        else:
            ax.scatter(scatter_x[ix], scatter_y[ix], scatter_z[ix], c=cdict[g], label = g, s = 5)
        #ax.scatter(scatter_x[ix], scatter_y[ix], c=cdict[g], label = g, s = 5)
    ax.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')
    fig.savefig(savepath+'.png')


def lof_score_visualize(y_te, y_pred_score,savepath):
    plt.clf()
    inlier_score = []
    outlier_score = []
    for index, label in enumerate(y_te):
        if y_te[index] == -1:
            outlier_score.append(y_pred_score[index])
        else:
            inlier_score.append(y_pred_score[index])

    inlier_y = [1 for i in range(len(inlier_score))]
    outlier_y = [2 for i in range(len(outlier_score))]

    plt.plot(inlier_score, inlier_y, 'o', label='inlier')
    plt.plot(outlier_score, outlier_y, 'x', label='outlier')
    plt.savefig(savepath+'outlierscore.png')

def metrix_change_visualize(pre, rec, acc, fpr, threshold, savepath):
    plt.clf()
    line1 = plt.plot(threshold, pre, color='blue', label="precision")
    line2 = plt.plot(threshold, rec, color='orange', label="recall")
    line3 = plt.plot(threshold, acc, color='green', label="accuracy")
    line4 = plt.plot(threshold, fpr, color='red', label='fp_rate')

    for x, y in zip(threshold, pre):
        label = "{:.2f}".format(y)
        plt.annotate(label, (x, y), textcoords='offset points', xytext=(0, 10), ha='center')
    for x, y in zip(threshold, rec):
        label = "{:.2f}".format(y)
        plt.annotate(label, (x, y), textcoords='offset points', xytext=(0, 10), ha='center')
    for x, y in zip(threshold, acc):
        label = "{:.2f}".format(y)
        plt.annotate(label, (x, y), textcoords='offset points', xytext=(0, 10), ha='center')
    for x, y in zip(threshold, fpr):
        label = "{:.2f}".format(y)
        plt.annotate(label, (x, y), textcoords='offset points', xytext=(0, 10), ha='center')

    plt.xlabel('Threshold')
    plt.ylabel('Performance')

    plt.legend(handles = [line1[0], line2[0], line3[0], line4[0]])

    plt.savefig(savepath+'performance_change.png')


