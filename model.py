import torch
import torch.nn as nn
from torch.nn import Module
import torch.nn.functional as F
import math
from torch.nn.parameter import Parameter

'''
class Attention(nn.Module):
    def __init__(self,hidden_size):
        super(Attention, self).__init__()

        self.project = nn.Sequential(
            nn.Linear(hidden_size, 1, bias=False)
        )

    def forward(self, z):
        w = self.project(z)
        beta = torch.softmax(w, dim=1)
        return (beta * z).sum(1), beta
'''

class GraphConvolution(Module):
    """
    A Graph Convolution Layer (GCN)
    """

    def __init__(self, in_features, out_features):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.weight)

    def forward(self, input_feature, adj):
        support = torch.mm(input_feature, self.weight)
        output = torch.spmm(adj, support)
        return output



class DGCN(nn.Module):

    def __init__(self, nfeat, nhid1,  nclass, q, k, dropout):
        super(DGCN, self).__init__()

        self.gc1 = GraphConvolution(nfeat, nhid1)
        self.gc2 = GraphConvolution(nhid1, nclass)
        self.lin = nn.Linear(nfeat, nclass)
        self.gcn = GraphConvolution(nclass, nclass)
        self.dropout = dropout
        self.q = q
        self.k = k
        self.apha = torch.nn.Parameter(torch.FloatTensor(1), requires_grad=True)
    def forward(self, x, adj):
        x = F.dropout(x, self.dropout, training=self.training)
        x1 = F.relu(self.gc1(x, adj))
        x1 = F.dropout(x1, self.dropout, training=self.training)
        x2 = F.softmax(self.gc2(x1, adj))
        wave_x = F.softmax(self.lin(x))
        apha = F.sigmoid(self.apha)

        emb1 = torch.add(x2, self.gcn(wave_x, self.k))
        emb2 = torch.add(x2, self.gcn(wave_x,self.q))
        emb = apha*emb1+(1-apha)*emb2

        return emb


def get_model(model_opt, nfeat, nhid1, nclass, q, k, dropout, cuda=True):
    if model_opt == "DGCN":
        model = DGCN(
            nfeat=nfeat,
            nhid1=nhid1,
            nclass=nclass,
            q=q,
            k=k,
            dropout=dropout
        )
        
    else:
        raise NotImplementedError('model:{} is not implemented!'.format(model_opt))

    if cuda: model.cuda()
    return model

