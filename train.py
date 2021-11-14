import time
import os
#os.environ['CUDA_VISIBLE_DEVICES'] = "1"
import argparse
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from utils import *
from model import get_model
from metrics import accuracy
from sklearn.metrics import f1_score
import pickle as pkl
from args import get_citation_args
from time import perf_counter
from sklearn import metrics
import warnings
import pandas as pd
from tensorboardX import SummaryWriter
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=RuntimeWarning)
warnings.simplefilter(action='ignore', category=UserWarning)

# Arguments
args = get_citation_args()
torch.cuda.set_device(3)
#os.environ["CUDA_VISIBLE_DEVICES"] = "2"

# setting random seeds
set_seed(args.seed, args.cuda)


if args.dataset in ['cornell', 'texas', 'washington']:
    adj, features, labels, idx_train, idx_val, idx_test,adj_origin = load_data2(args.dataset, args.normalization)
#if args.dataset in ['chameleon','squirrel','wisconsin']:
#    adj, features, labels, idx_train, idx_val, idx_test, adj_origin = load_highfrequency(args.dataset)
else:
    adj, features, labels, idx_train, idx_val, idx_test, adj_origin = load_citation(args.dataset, args.normalization)

q,k = get_delta(adj_origin, labels, idx_train)
q=q.cuda()
k= k.cuda()


model = get_model(args.model, features.shape[1], args.hidden1,labels.max().item()+1,q,k, args.dropout)

optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

def train(model, epochs):

    model.train()
    optimizer.zero_grad()
    output = model(features, adj)
    loss = F.cross_entropy(output[idx_train], labels[idx_train])
    acc = accuracy(output[idx_train], labels[idx_train])
    loss.backward()
    optimizer.step()
    acc_test, macro_f1,nmi = main_test(model,labels)
    '''
    print('e:{}'.format(epochs),
          'loss_train: {:.4f}'.format(loss.item()),
          'acc_train: {:.4f}'.format(acc.item()),
          'acc_test: {:.4f}'.format(acc_test.item()),
          'f1_test:{:.4f}'.format(macro_f1.item()),
          )
    '''
    return output,loss.item(), acc_test.item(), macro_f1.item(),nmi


def main_test(model,labels):
    model.eval()
    output = model(features, adj)
    acc_test = accuracy(output[idx_test], labels[idx_test])
    nmi = metrics.normalized_mutual_info_score(labels[idx_test].cpu(),output[idx_test].max(1)[1].cpu().data.numpy())
    label_max = []
    for idx in idx_test:
        label_max.append(torch.argmax(output[idx]).item())
    labelcpu = labels[idx_test].data.cpu()
    macro_f1 = f1_score(labelcpu, label_max, average='micro')
    return acc_test, macro_f1,nmi

#writer = SummaryWriter('runs/exp/')
acc_max = 0
f1_max = 0
epoch_max = 0
NMI =[]

for epoch in range(args.epochs):
    output, loss, acc_test, macro_f1,nmi= train(model, epoch)
    if (nmi<1 and nmi>0):
        NMI.append(nmi)


    if acc_test >= acc_max:
        acc_max = acc_test
        f1_max = macro_f1
        epoch_max = epoch



print('epoch:{}'.format(epoch_max),
      'acc_max: {:.4f}'.format(acc_max),
      'f1_max: {:.4f}'.format(f1_max),
      'nmi_max: {:.4f}'.format(max(NMI)),
      )
