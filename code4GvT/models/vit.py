# https://github.com/lucidrains/vit-pytorch/blob/main/vit_pytorch/vit_pytorch.py

import torch
import torch.nn.functional as F
from einops import rearrange
from torch import nn
from functools import partial
from itertools import repeat
from collections import OrderedDict
from einops import rearrange
from einops.layers.torch import Rearrange
import math
from pygcn.models import GCN
# from pygcn.models_gru import GCN_gru
import numpy as np
# import cupy as cp
# from models import mobilenet

from models import *
from torch.autograd import Variable
dtype = torch.cuda.FloatTensor

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# from torch._six import container_abcs

# From PyTorch internals
def _ntuple(n):
    def parse(x):
        # if isinstance(x, container_abcs.Iterable):
        #     return x
        return tuple(repeat(x, n))

    return parse


to_2tuple = _ntuple(2)

MIN_NUM_PATCHES = 16


class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, *args, **kwargs):
        return self.fn(x, *args, **kwargs) + x


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, *args, **kwargs):
        temp = self.norm(x)
        return self.fn(temp, *args, **kwargs)


# class PreForward(nn.Module):
#     def __init__(self, dim, hidden_dim, kernel_size, num_channels, dropout=0.):
#         super().__init__()
#         # self.net = nn.Sequential(
#         #     nn.Linear(dim, hidden_dim),
#         #     nn.GELU(),
#         #     nn.Dropout(dropout),
#         #     nn.Linear(hidden_dim, dim),
#         #     nn.Dropout(dropout)
#         # )
#         self.tcn = TemporalConvNet(dim, num_channels, hidden_dim, kernel_size, dropout)
#         # self.net = nn.Sequential(
#         #     nn.Linear(dim, hidden_dim),
#         #     nn.GELU(),
#         #     nn.Dropout(dropout),
#         # )
#
#     def forward(self, x):
#         r = self.tcn(x.permute(0, 2, 1)).permute(0, 2, 1)
#         # r = self.net(r)
#         return r


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, image_size, patch_size, kernel_size, dropout=0.):
        super().__init__()
        # self.net = nn.Sequential(
        #     nn.Linear(dim, hidden_dim),
        #     nn.GELU(),
        #     nn.Dropout(dropout),
        #     nn.Linear(hidden_dim, dim),
        #     nn.Dropout(dropout)
        # )
        self.net = nn.Identity()
    def forward(self, x):
        return self.net(x)


# def inverse_gumbel_cdf(y, mu, beta):
#     return mu - beta * torch.log(-torch.log(y))


class Attention(nn.Module):
    def __init__(self,
                 dim,
                 image_size,
                 patch_size,
                 heads=8,
                 dropout=0,
                 qkv_bias=False,
                 attn_drop=0.,
                 proj_drop=0.,
                 downsample=0.,
                 kernel_size=3,
                 stride_kv=1,
                 stride_q=1,
                 padding_kv=1,
                 padding_q=1,
                 with_cls_token=False
                 ):
        super().__init__()
        self.scale = dim ** -0.5
        self.drop_ratio = 0.1

        # self.to_qkv = nn.Linear(dim, dim * 3, bias = False)
        # self.to_out = nn.Sequential(
        #     nn.Linear(dim, dim),
        #     nn.Dropout(dropout)
        # )

        self.stride_kv = stride_kv
        self.stride_q = stride_q
        self.dim = dim
        self.num_heads = heads
        self.with_cls_token = with_cls_token

        self.length = int(image_size / patch_size) ** 2
        self.length2 = int(image_size / patch_size * downsample) ** 2

        dim_in = dim
        dim_out = dim


        self.conv_proj_q = GCN(nfeat=dim_in,
                               head_num=heads,
                               nhid=dim_out,
                               image_size=image_size,
                               patch_size=patch_size,
                               stride=2,
                               padding=1,  # using 2 when kernel_size = 4
                               kernel_size=kernel_size,  # kernel_size of GCN
                               nclass=None,
                               dropout=dropout)
        self.conv_proj_k = GCN(nfeat=dim_in,
                               head_num=heads,
                               nhid=dim_out,
                               image_size=image_size,
                               patch_size=patch_size,
                               stride=2,
                               padding=1,  # using 2 when kernel_size = 4
                               kernel_size=kernel_size,  # kernel_size of GCN
                               nclass=None,
                               dropout=dropout)
        self.conv_proj_v = GCN(nfeat=dim_in,  # GCN_gru
                                   head_num=heads,
                                   nhid=dim_out,
                                   image_size=image_size,
                                   patch_size=patch_size,
                                   stride=2,
                                   padding=1,  # using 2 when kernel_size = 4
                                   kernel_size=kernel_size,  # kernel_size of GCN
                                   nclass=None,
                                   dropout=dropout)

        self.proj_q = nn.Linear(dim_in, dim_out, bias=qkv_bias)
        self.proj_k = nn.Linear(dim_in, dim_out, bias=qkv_bias)
        self.proj_v = nn.Linear(dim_in, dim_out, bias=qkv_bias)



        self.attn_drop = nn.Dropout(attn_drop)  #
        self.proj_drop = nn.Dropout(proj_drop)  #
        self.leakyrelu = nn.LeakyReLU()



        sparse_D = torch.ones((self.num_heads, self.length), requires_grad=True).to(device)
        self.sparse_D = torch.nn.Parameter(sparse_D)
        self.register_parameter("sparse_D", self.sparse_D)

        sparse_D2 = torch.ones((self.num_heads, self.length2), requires_grad=True).to(device)
        self.sparse_D2 = torch.nn.Parameter(sparse_D2)
        self.register_parameter("sparse_D2", self.sparse_D2)

        randomatrix = torch.randn((int(self.num_heads),
                                   int(self.num_heads)), requires_grad=True).to(device)
        self.randomatrix = torch.nn.Parameter(randomatrix)
        self.register_parameter("Ablah", self.randomatrix)

        # from torch.nn.parameter import Parameter
        # self.randomatrix = Parameter(torch.FloatTensor(int(self.num_heads), int(self.num_heads))).to(device)

        self.proj_k_f = nn.Linear(2 * dim_in, self.num_heads, bias=qkv_bias)
        self.proj_k_f2 = nn.Linear(self.num_heads, self.length, bias=qkv_bias)
        self.proj_k_f3 = nn.Linear(self.num_heads, self.length2, bias=qkv_bias)

    def forward_conv_qk(self, x, rep_adj):
        # if self.with_cls_token:
        #     cls_token, x = torch.split(x, [1, h * w], 1)
        if self.conv_proj_q is not None:
            q = self.conv_proj_q(x, rep_adj)
            # q = F.dropout(F.relu(q), self.drop_ratio, training=self.training)
        else:
            q = rearrange(x, 'b c h w -> b (h w) c')

        if self.conv_proj_k is not None:
            k = self.conv_proj_k(x, rep_adj)
            # k = F.dropout(F.relu(k), self.drop_ratio, training=self.training)
        else:
            k = rearrange(x, 'b c h w -> b (h w) c')

        return q, k

    def forward_conv_v(self, x, rep_adj):
        # if self.with_cls_token:
        #     cls_token, x = torch.split(x, [1, h * w], 1)

        if self.conv_proj_v is not None:
            v = self.conv_proj_v(x, rep_adj)
            # v = F.dropout(v, self.drop_ratio, training=self.training)
        else:
            v = rearrange(x, 'b c h w -> b (h w) c')

        return v

    def forward(self, x, adj, label):
        x = rearrange(x, 'b t (h d) -> b h t d', h=self.num_heads)
        current_length = adj.size(-1)
        b, head, _, _ = x.size()

        if (
                self.conv_proj_q is not None
                or self.conv_proj_k is not None
                or self.conv_proj_v is not None
        ):
            q, k = self.forward_conv_qk(x, adj)
        q = (self.proj_q(rearrange(q, 'b h t d -> b t (h d)', h=head)))
        k = (self.proj_k(rearrange(k, 'b h t d -> b t (h d)', h=head)))

        qk = torch.concat((q, k), dim=-1)

        Random_RM0 = F.gelu(self.proj_k_f(qk))
        Random_RM = torch.matmul(Random_RM0.permute(0, 2, 1), Random_RM0)

        if label == 0:
            Random_RM = torch.sigmoid(self.proj_k_f2(Random_RM) / self.sparse_D)
        else:
            Random_RM = torch.sigmoid(self.proj_k_f3(Random_RM) / self.sparse_D2)

        Random_RM = torch.diag_embed(Random_RM)

        k = rearrange(k, 'b t (h d) -> b h t d', h=head)
        q = rearrange(q, 'b t (h d) -> b h t d', h=head)
        attn_score = torch.einsum('bhlk,bhtk->bhlt', [q, k]) * self.scale

        attn_score = self.leakyrelu(attn_score)  # F.gelu

        ## ---1-----
        attn_score = torch.matmul(torch.matmul(Random_RM, attn_score), Random_RM)
        ## ---2-----
        Lambda = self.randomatrix
        Lambda = Lambda.expand(b, -1, -1).to(device)
        attn_score = rearrange(attn_score, 'b h l t -> b h (l t)')
        attn_score = torch.einsum('blh,bhk->blk', [Lambda, attn_score])
        attn_score = rearrange(attn_score, 'b h (l k) -> b h l k', l=current_length)

        zero_vec = -1e12 * torch.ones_like(attn_score)  # 将没有连接的边置为负无穷
        attn_score = torch.where(adj > 0, attn_score, zero_vec)

        # m_r = torch.ones_like(attn_score) * 0.1
        # attn_score = attn_score + torch.bernoulli(m_r)*-1e12

        attn_score = F.softmax(attn_score, dim=-1)
        # attn_score = self.attn_drop(attn_score)

        rep_adj = attn_score

        # rep_adj = similarity.matmul(rep_adj).matmul(similarity).to(device)
        # rep_adj = pointer_diag.matmul(rep_adj).matmul(pointer_diag).to(device)

        v = self.forward_conv_v(x, rep_adj)

        v = F.gelu(self.proj_v(rearrange(v, 'b h t d -> b t (h d)', h=head)))

        out = self.proj_drop(v)

        return out


class ConvEmbed(nn.Module):
    """ Image to Conv Embedding

    """

    def __init__(self,
                 image_size,
                 patch_size,
                 kernel_size,
                 batch_size,
                 in_chans,
                 embed_dim,
                 stride,
                 padding,
                 norm_layer=None):
        super().__init__()
        # kernel_size = to_2tuple(kernel_size)
        # self.patch_size = patch_size

        # self.proj = ResNet18(embed_dim).to(device)

        # self.proj = nn.Conv2d(
        #     in_chans, embed_dim,
        #     kernel_size=kernel_size,
        #     stride=stride,
        #     padding=padding
        # )

        self.proj = nn.Sequential(OrderedDict([
            ('conv1', nn.Conv2d(
                in_chans, int(embed_dim),
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                # groups=in_chans
            )),
            #### ('pooling', nn.AdaptiveMaxPool2d((int(image_size / patch_size), int(image_size / patch_size)))),
            ('bn', nn.BatchNorm2d(int(embed_dim))),
            ('relu', nn.GELU()),
            ('conv2', nn.Conv2d(
                int(embed_dim), int(embed_dim),
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                groups=int(embed_dim)
            )),
            ('pooling', nn.AdaptiveMaxPool2d((int(patch_size/7), int(patch_size/7)))),
            # ('bn', nn.BatchNorm2d(int(embed_dim))),
            # ('relu', nn.GELU()),
        ]))

        # self.proj = nn.Sequential(OrderedDict([
        #     ('conv1', nn.Conv2d(
        #         in_chans, int(embed_dim),
        #         kernel_size=kernel_size,
        #         stride=stride,
        #         padding=padding,
        #         # groups=in_chans
        #     )),
        #     ('pooling', nn.AdaptiveMaxPool2d((int(image_size / patch_size / 4), int(image_size / patch_size / 4)))),
        #     ('bn', nn.BatchNorm2d(int(embed_dim))),
        #     ('relu', nn.GELU()),
        # ]))
        # self.norm = norm_layer(embed_dim) if norm_layer else None

    def forward(self, x):
        sp_features = self.proj(x).to(device)  # proj_conv  proj

        return sp_features


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, mlp_dim, dropout, image_size, patch_size, kernel_size, downsample, batch_size, in_chans,
                 patch_stride, patch_padding, norm_layer=nn.LayerNorm):
        super().__init__()

        self.patch_embed = ConvEmbed(
            image_size=image_size,
            patch_size=patch_size,
            kernel_size=kernel_size,
            batch_size=batch_size,
            in_chans=in_chans,
            stride=patch_stride,
            padding=patch_padding,
            embed_dim=dim,
            norm_layer=norm_layer
        )
        self.patch_dim = ((patch_size//7)**2) * int(dim)
        self.dim = dim
        # channels = 3
        # self.patch_dim = channels * patch_size ** 2

        self.patch_to_embedding = nn.Linear(self.patch_dim, dim).to(device)

        self.layers = nn.ModuleList([])
        self.depth = depth
        self.depth4pool = depth//2
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                FeedForward(dim, mlp_dim, image_size, patch_size, kernel_size, dropout=dropout),
                Residual(PreNorm(dim, Attention(dim, image_size=image_size, patch_size=patch_size, heads=heads, dropout=dropout, downsample=downsample, kernel_size=kernel_size))),
            ]))

        self.dropout = nn.Dropout(dropout)

        # self.norm = nn.ModuleList([])
        # for _ in range(depth):
        #     self.norm.append(nn.LayerNorm(dim))

        self.patch_size = patch_size
        self.batch_size = batch_size
        self.head_num = heads
        # UT = torch.randn((int(image_size/patch_size*downsample)**2, dim), requires_grad=True).to(device)
        # self.UT = torch.nn.Parameter(UT)
        # self.register_parameter("Ablah2", self.UT)

        self.Upool = nn.Sequential(
            nn.Linear(dim, int(image_size/patch_size*downsample)**2, bias=True),
            # nn.Dropout(dropout)
        )

        self.Upool_out = nn.Sequential(
            nn.Linear(dim, 1, bias=True),
        )

    def forward(self, img, adj):
        p = self.patch_size
        b, n, imgh, imgw = img.shape

        x = rearrange(img, 'b c (h p1) (w p2) -> (b h w) (c) (p1) (p2)', p1=p, p2=p)
        conv_img = self.patch_embed(x)
        conv_img = rearrange(conv_img, '(b s) c p1 p2 -> b s (c p1 p2)', b=b)
        x = self.patch_to_embedding(conv_img)

        # x = rearrange(img, 'b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=p, p2=p)
        # x = self.patch_to_embedding(x)

        x = rearrange(x, 'b (h w) c -> b c h w', h=int(imgh / p), w=int(imgw / p))
        x = rearrange(x, 'b c h w -> b (h w) c')

        rep_adj = adj.expand(b, self.head_num, -1, -1).to(device)

        index = 0
        for pre, attn in self.layers:
            x = pre(x)
            # x = attn(x, self.rep_adj, 0)
            if index < self.depth4pool:
                x = attn(x, rep_adj, 0)
            else:
                if index == self.depth4pool:
                    # temp = torch.matmul(self.UT, x.permute(0, 2, 1))
                    temp = self.Upool(x).permute(0, 2, 1)
                    temp = F.gumbel_softmax(temp, dim=-1, tau=1.0)
                    x = torch.matmul(temp, x)
                    C = temp
                    C = C.unsqueeze(dim=1).expand(b, self.head_num, -1, -1).to(device)
                    temp2 = torch.matmul(C, rep_adj)
                    rep_adj = torch.matmul(temp2, C.permute(0, 1, 3, 2))

                att = attn(x, rep_adj, 1)
                x = att

            index = index + 1
        temp = self.Upool_out(x).permute(0, 2, 1)
        temp = F.gumbel_softmax(temp, dim=-1, tau=1.0)
        # temp = F.softmax(temp/2)
        x = torch.matmul(temp, x)

        x_out = x
        # x_out = F.normalize(x, dim=-1)
        return x_out


class ViT(nn.Module):
    def __init__(self, *, image_size, patch_size, kernel_size, downsample, batch_size, num_classes, dim, depth, heads,
                 mlp_dim, patch_stride, patch_pading, in_chans, dropout=0., emb_dropout=0., expansion_factor=1):
        super().__init__()
        assert image_size % patch_size == 0, 'image dimensions must be divisible by the patch size'
        pantchesalow = image_size // patch_size
        num_patches = pantchesalow ** 2

        adj_matrix = [[0 for i in range(num_patches)] for i in range(num_patches)]
        adj_matrix = torch.as_tensor(adj_matrix).float().to(device)

        for j in range(num_patches):
            if (j - pantchesalow - 1) >= 0:
                adj_matrix[j][j - 1] = 1
                adj_matrix[j][j - pantchesalow] = 1
                adj_matrix[j][j - pantchesalow-1] = 1
                adj_matrix[j][j - pantchesalow+1] = 1
            if (j + pantchesalow + 1) < num_patches:
                adj_matrix[j][j + 1] = 1
                adj_matrix[j][j + pantchesalow] = 1
                adj_matrix[j][j + pantchesalow-1] = 1
                adj_matrix[j][j + pantchesalow+1] = 1
        self.adj_matrix = adj_matrix

        self.patch_size = patch_size

        # self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        # self.patch_to_embedding = nn.Linear(patch_dim, dim)
        # self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, mlp_dim, dropout, image_size, patch_size, kernel_size, downsample,
                                       batch_size, in_chans, patch_stride=patch_stride, patch_padding=patch_pading)

        self.to_cls_token = nn.Identity()

        self.predictor = Predictor(dim, num_classes)

    def forward(self, img):
        x = self.transformer(img, self.adj_matrix)
        # x = self.to_cls_token(x[:, -1])
        # pred = self.to_cls_token(x.squeeze())
        class_result = self.predictor(x.squeeze())
        return class_result


# class MLP(nn.Module):
#     def __init__(self, dim, projection_size):
#         super().__init__()
#         hidden_size = dim*2
#         self.net = nn.Sequential(
#             nn.Linear(dim, hidden_size),
#             nn.BatchNorm1d(hidden_size),
#             nn.ReLU(inplace=True),
#             nn.Linear(hidden_size, projection_size)
#         )
#
#     def forward(self, x):
#         return self.net(x)


class Predictor(nn.Module):
    def __init__(self, dim, num_classes):
        super().__init__()
        hidden_size = dim*2
        self.mlp_head = nn.Sequential(
            nn.Linear(dim, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, num_classes)
        )

    def forward(self, x):
        return self.mlp_head(x.unsqueeze(0))  #.unsqueeze(0)
