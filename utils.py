import os
import numpy as np
import scipy.sparse as sp
from scipy.sparse import csr_matrix
import torch
import sys
import pickle as pkl
import networkx as nx
from normalization import fetch_normalization, row_normalize
from time import perf_counter
import math
import sklearn.preprocessing as preprocess
import random
import scipy.io as sio
import h5py


def parse_index_file(filename):
    """Parse index file."""
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index

def preprocess_citation(adj, features, normalization="AugNormAdj"):
    adj_normalizer = fetch_normalization(normalization)
    adj = adj_normalizer(adj)
    features = row_normalize(features)
    return adj, features

def preprocess_adj(adj, normalization="AugNormAdj"):
    adj_normalizer = fetch_normalization(normalization)
    adj = adj_normalizer(adj)
  
    return adj
def normalize_(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1),dtype=np.float32)
    rowsum = (rowsum==0)*1+rowsum
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx
def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)

def sys_normalized_adjacency(adj):
    adj = sp.coo_matrix(adj)
    adj = adj + sp.eye(adj.shape[0])
    row_sum = np.array(adj.sum(1))
    row_sum = (row_sum == 0) * 1 + row_sum
    d_inv_sqrt = np.power(row_sum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    adj = d_mat_inv_sqrt.dot(adj).dot(d_mat_inv_sqrt)

    return adj.tocoo()
def load_data2(dataset, normalization="AugNormAdj", cuda=False):
    filename = dataset
    rootdir = './data/'
    '''
    out = open(rootdir + filename + '/label.txt', 'w')#把顺序排列的节点标签乱序，存放于label.txt
    lines = []
    with open(rootdir + filename + '/labels.txt', 'r') as infile:#顺序标签
        for line in infile:
            lines.append(line)
        random.shuffle(lines)
        for line in lines:
            out.write(line)
    infile.close()
    out.close()
    '''
    f = open(rootdir + filename + '/label.txt', 'r')#打乱顺序后的标签
    label = []
    train_index = []
    for line in f.readlines():
        line = line.split()
        label.append(int(line[1]))#第一列为NodeId
        train_index.append(int(line[0]))#对应节点的label
    f.close()

    
    f = open(rootdir + filename + '/labels.txt', 'r')#未打乱顺序的标签，原标签文件，节点按顺序排列
    labels = []
    for line in f.readlines():
        line = line.split()
        labels.append(int(line[1]))
    f.close()

    labelset = np.unique(labels)
    labeldict = dict(zip(labelset, range(len(labelset))))
    labels = np.array([labeldict[x] for x in labels])
    # print(labels)
    lset = np.unique(label)
    ldict = dict(zip(lset, range(len(lset))))
    label = np.array([ldict[x] for x in label])

    try:
        data_c = h5py.File(rootdir + filename + '/' + filename + '_content_n.mat')['content'][:].T
    except BaseException:
        data_c = sio.loadmat(rootdir + filename + '/' + filename + '_content_n.mat')['content']
    # data_c = h5py.File(rootdir+filename+'/'+filename+'_content_n.mat')['content'][:].T
    data_t = h5py.File(rootdir + filename + '/' + filename + '_n.mat')['G'][:].T
    #data_l = h5py.File(rootdir + filename + '/' + filename + '_rlabels_n.mat')['labels'][:].T

    features = torch.FloatTensor(data_c)
    #labels = data_l.T
    #labels = labels[0]
    adj = csr_matrix(data_t)
    adj_origin = adj
    adj = preprocess_adj(adj, normalization)

    n = labels.shape[0]
    #idx = [i for i in range(n)]
    # random.shuffle(idx)
    r0 = int(n * 0.6)
    r1 = int(n * 0.6)
    r2 = int(n * 0.8)

    #idx_train = np.array(idx[:r0])
    #idx_val = np.array(idx[r1:r2])
    #idx_test = np.array(idx[r2:])

    adj_train_index = train_index[:r0]
    adj_test_index = train_index[r2:]
    adj_val_index = train_index[r1:r2]

    labels = torch.LongTensor(labels)

    adj = sparse_mx_to_torch_sparse_tensor(adj).float()
    adj_origin = torch.LongTensor(np.array(adj_origin.todense()))

    adj_train_index = torch.LongTensor(adj_train_index)
    adj_val_index = torch.LongTensor(adj_val_index)
    adj_test_index = torch.LongTensor(adj_test_index)

    if cuda:
        features = features.cuda()
        adj = adj.cuda()
        labels = labels.cuda()
        adj_train_index = adj_train_index.cuda()
        adj_test_index = adj_test_index.cuda()
        adj_val_index = adj_val_index.cuda()
        adj_origin = adj_origin.cuda()

    return adj, features, labels,  adj_train_index,adj_val_index, adj_test_index,adj_origin

def load_highfrequency(dataset_str="cornell",cuda=True):
    graph_adjacenct_list_file_path = 'high_freq/{}/out1_graph_edges.txt'.format(dataset_str)
    graph_node_features_and_labels_file_path = 'high_freq/{}/out1_node_feature_label.txt'.format(dataset_str)

    G = nx.DiGraph()
    graph_node_features_dict = {}
    graph_labels_dict = {}

    with open(graph_node_features_and_labels_file_path) as graph_node_features_and_labels_file:
        graph_node_features_and_labels_file.readline()
        for line in graph_node_features_and_labels_file:
            line = line.rstrip().split('\t')
            assert (len(line) == 3)
            assert (int(line[0]) not in graph_node_features_dict and int(line[0]) not in graph_labels_dict)
            graph_node_features_dict[int(line[0])] = np.array(line[1].split(','), dtype=np.uint8)
            graph_labels_dict[int(line[0])] = int(line[2])

    with open(graph_adjacenct_list_file_path) as graph_adjacenct_list_file:
        graph_adjacenct_list_file.readline()
        for line in graph_adjacenct_list_file:
            line = line.rstrip().split('\t')
            assert (len(line) == 2)
            if int(line[0]) not in G:
                G.add_node(int(line[0]), features=graph_node_features_dict[int(line[0])],
                           label=graph_labels_dict[int(line[0])])
            if int(line[1]) not in G:
                G.add_node(int(line[1]), features=graph_node_features_dict[int(line[1])],
                           label=graph_labels_dict[int(line[1])])
            G.add_edge(int(line[0]), int(line[1]))

    adj = nx.adjacency_matrix(G, sorted(G.nodes()))
    graph = nx.DiGraph(adj)
    # print(adj.nnz)
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    adj_origin = adj
    features = np.array([features for _, features in sorted(G.nodes(data='features'), key=lambda x: x[0])])
    labels = np.array([label for _, label in sorted(G.nodes(data='label'), key=lambda x: x[0])])

    features = normalize_(features)

    n = len(labels.tolist())
    idx = [i for i in range(n)]
    r0 = int(n * 0.6)
    r1 = int(n * 0.6)
    r2 = int(n * 0.8)
    train = np.array(idx[:r0])
    val = np.array(idx[r1:r2])
    test = np.array(idx[r2:])

    nclass = len(set(labels.tolist()))
    features = torch.FloatTensor(features)
    labels = torch.LongTensor(labels)
    train = torch.LongTensor(train)
    val = torch.LongTensor(val)
    test = torch.LongTensor(test)
    adj_origin = torch.FloatTensor(np.array(adj_origin.todense()))
    adj = sys_normalized_adjacency(adj)

    adj = sparse_mx_to_torch_sparse_tensor(adj)
    if cuda:
        features = features.cuda()
        adj = adj.cuda()
        labels = labels.cuda()
        train = train.cuda()
        val = val.cuda()
        test = test.cuda()
        adj_origin = adj_origin.cuda()


    return adj, features, labels, train, val, test, adj_origin

def load_citation(dataset_str="cora", normalization="AugNormAdj", cuda=True,task_type = "full"):
    """
    Load Citation Networks Datasets.
    """
    names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']
    objects = []

    for i in range(len(names)):
        with open("data/ind.{}.{}".format(dataset_str.lower(), names[i]), 'rb') as f:
            if sys.version_info > (3, 0):
                objects.append(pkl.load(f, encoding='latin1'))
            else:
                objects.append(pkl.load(f))

    x, y, tx, ty, allx, ally, graph = tuple(objects)
    test_idx_reorder = parse_index_file("data/ind.{}.test.index".format(dataset_str))
    test_idx_range = np.sort(test_idx_reorder)

    if dataset_str == 'citeseer':
        # Fix citeseer dataset (there are some isolated nodes in the graph)
        # Find isolated nodes, add them as zero-vecs into the right position
        test_idx_range_full = range(min(test_idx_reorder), max(test_idx_reorder)+1)
        tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))
        tx_extended[test_idx_range-min(test_idx_range), :] = tx
        tx = tx_extended
        ty_extended = np.zeros((len(test_idx_range_full), y.shape[1]))
        ty_extended[test_idx_range-min(test_idx_range), :] = ty
        ty = ty_extended

    features = sp.vstack((allx, tx)).tolil()
    features[test_idx_reorder, :] = features[test_idx_range, :]
    adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    adj_origin = adj
    labels = np.vstack((ally, ty))
    labels[test_idx_reorder, :] = labels[test_idx_range, :]



    if task_type == "full":
        print("Load full supervised task.")
        # supervised setting
        idx_test = test_idx_range.tolist()
        idx_train = range(len(ally) - 500)
        idx_val = range(len(ally) - 500, len(ally))
    elif task_type == "semi":
        print("Load semi-supervised task.")
        # semi-supervised setting
        idx_test = test_idx_range.tolist()
        idx_train = range(len(y))
        idx_val = range(len(y), len(y) + 500)


    adj, features = preprocess_citation(adj, features, normalization)

    features = torch.FloatTensor(np.array(features.todense())).float()
    labels = np.argmax(labels, axis=1)
    labels = torch.LongTensor(labels)
    adj = sparse_mx_to_torch_sparse_tensor(adj).float()
    adj_origin = torch.LongTensor(np.array(adj_origin.todense()))
    idx_train = torch.LongTensor(idx_train)
    idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)
    if cuda:
        features = features.cuda()
        adj = adj.cuda()   
        labels = labels.cuda()
        idx_train = idx_train.cuda()
        idx_val = idx_val.cuda()
        idx_test = idx_test.cuda()
        adj_origin = adj_origin.cuda()

    return adj, features, labels, idx_train, idx_val, idx_test,adj_origin

def set_seed(seed, cuda):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if cuda: torch.cuda.manual_seed(seed)

def normalized_adjacency(adj):
   #adj = adj + sp.eye(adj.shape[0])
   adj = sp.coo_matrix(adj)
   row_sum = np.array(adj.sum(1))
   d_inv_sqrt = np.power(row_sum, -0.5).flatten()
   d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
   d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
   return d_mat_inv_sqrt.dot(adj).dot(d_mat_inv_sqrt).tocoo()
   

def get_delta(adj_org,labels, idx_train):
    adj_org = torch.tensor(adj_org.cpu(), dtype=torch.double)  # 不加自环
    adj = normalized_adjacency(adj_org)
    adj = sparse_mx_to_torch_sparse_tensor(adj).float()
    adj = adj.to_dense() #不加自环 经过归一化的
    adj = torch.tensor(adj.cpu(), dtype=torch.double)

    wMax = torch.zeros(adj.shape, dtype=torch.double)
    bMax = torch.zeros(adj.shape, dtype=torch.double)
   
    for i, e1 in enumerate(idx_train):
        for j, e2 in enumerate(idx_train):
            if adj_org[e1, e2] == 1:  # 有边相连
                if labels[e1] != labels[e2]:#相连的两个节点属于不同的class
                    bMax[e1, e2] = -1
                if labels[e1] == labels[e2]:#相连的两个节点来自同一个class
                    wMax[e1, e2] = 1

    q = torch.mul(adj, bMax)
    q = torch.tensor(q, dtype=torch.float32)

    k = torch.mul(adj, wMax)
    k = torch.tensor(k, dtype=torch.float32)
    return q,k

