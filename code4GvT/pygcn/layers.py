import math
import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np
from einops import rearrange
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
# from models import resnet4GCN
from collections import OrderedDict
device = 'cuda' if torch.cuda.is_available() else 'cpu'


class GraphConvolution(Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, head_num, out_features, image_size, patch_size, stride=1, padding=1, kernel_size=3, bias=True):
        super(GraphConvolution, self).__init__()
        in_features = int(in_features/head_num)
        out_features = int(out_features/head_num)
        self.image_size = image_size
        self.patch_size = patch_size

        # 选择在图卷积中使用全连接层
        # (1)
        self.weight = Parameter(torch.FloatTensor(head_num, in_features, out_features)).to(device)

        # (2)
        # self.head_num = head_num
        # self.weight = Parameter(torch.FloatTensor(int(in_features*head_num), int(out_features*head_num)))

        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features)).to(device)
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

        #选择在图卷积中全连接层替换为卷积操作
        # (3)
        # self.proj = nn.Sequential(OrderedDict([
        #     ('conv1', nn.Conv2d(
        #         in_features,
        #         out_features,
        #         kernel_size=kernel_size,
        #         padding=padding,
        #         stride=stride,
        #         bias=False,
        #         groups=in_features #  or   2 int(in_features/8)
        #     )),
        #     # ('bn', nn.BatchNorm2d(in_features)),
        #     # ('conv2', nn.Conv2d(
        #     #     out_features, out_features,
        #     #     kernel_size=1,
        #     # )),
        #     ('bn', nn.BatchNorm2d(in_features)),
        #     ('relu', nn.GELU()),
        # ]))
        # self.pooling = nn.AdaptiveAvgPool2d((self.image_size // self.patch_size, self.image_size // self.patch_size))

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def norm(self, adjcency):
        degree = torch.pow(torch.einsum('ihjk->ihj', [adjcency]), -0.5)

        degree_diag = torch.diag_embed(degree.squeeze())

        norm_adj = degree_diag.matmul(adjcency).matmul(degree_diag).to(device)
        return norm_adj  # norm_adj

    def forward(self, input, adj):
        # eye = torch.eye(adj.size(2))
        # eye = eye.expand(adj.size(1), -1, -1).to(device)
        # eye = eye.expand(adj.size(0), -1, -1, -1).to(device)
        # adj = adj + eye

        # 选择在图卷积中全连接层替换为卷积操作
        # (1)
        support = torch.matmul(input, self.weight)
        norm_adj = self.norm(adj)
        output = torch.matmul(norm_adj, support)  # spmm
        if self.bias is not None:
            output = output + self.bias

        return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(in_features) + ' -> ' \
               + str(out_features) + ')'
