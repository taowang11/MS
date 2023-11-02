import numpy as np
import torch
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, matthews_corrcoef
from model import *
from sklearn.metrics import average_precision_score
def metric(label, output):
    zs = torch.sigmoid(output).to('cpu').data.numpy()
    # print(zs.shape)
    ts = label.to('cpu').data.numpy()
    # print(ts.shape)
    preds = list(map(lambda x: (x >= 0.5).astype(int), zs))

    auc = roc_auc_score(ts, zs)
    aupr = average_precision_score(ts, zs)

    preds_list, t_list = [], []
    preds_list = np.append(preds_list, preds)
    t_list = np.append(t_list, ts)
    acc = accuracy_score(t_list, preds_list)
    precision = precision_score(t_list, preds_list)

    recall = recall_score(t_list, preds_list)
    mcc = matthews_corrcoef(t_list, preds_list)

    f1_scroe = (2 * precision * recall) / (recall + precision)
    return auc, acc, precision, recall, f1_scroe, mcc
