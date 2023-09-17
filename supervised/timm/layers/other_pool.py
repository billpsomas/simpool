import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn import init

from functools import partial
from typing import Optional, Tuple

from .otk.layers import OTKernel
from .gather_excite import GatherExcite


def prepare_input(self, x):
    if len(x.shape) == 3: # Transformer
        # Input tensor dimensions:
        # x: (B, N, d), where B is batch size, N are patch tokens, d is depth (channels)
        B, N, d = x.shape
        return x
    if len(x.shape) == 4: # CNN
        # Input tensor dimensions:
        # x: (B, d, H, W), where B is batch size, d is depth (channels), H is height, and W is width
        B, d, H, W = x.shape
        x = x.reshape(B, d, H*W).permute(0, 2, 1) # (B, d, H, W) -> (B, d, H*W) -> (B, H*W, d)
        return x
    else:
        raise ValueError(f"Unsupported number of dimensions in input tensor: {len(x.shape)}")


class LSEPool(nn.Module):
    """
    Learnable LSE pooling with a shared parameter
    """

    def __init__(self, r=10, learnable=True):
        super(LSEPool, self).__init__()
        if learnable:
            self.r = nn.Parameter(torch.ones(1) * r)
        else:
            self.r = r

    def forward(self, x):
        s = (x.size(2) * x.size(3))
        x_max = F.adaptive_max_pool2d(x, 1)
        exp = torch.exp(self.r * (x - x_max))
        sumexp = 1 / s * torch.sum(exp, dim=(2, 3))
        sumexp = sumexp.view(sumexp.size(0), -1, 1, 1)
        logsumexp = x_max + 1 / self.r * torch.log(sumexp)
        return logsumexp

class GEM(nn.Module):
    """
    GEM pooling
    """

    def __init__(self, gamma=2, kernel=(7, 7)):
        super(GEM, self).__init__()
        self.gamma = gamma
        self.kernel = kernel
        self.pool = torch.nn.LPPool2d(self.gamma, self.kernel)

    def forward(self, x):
        x = self.pool(x)
        return x

class GeneralizedMP(nn.Module):
    """ Generalized Max Pooling
    """
    def __init__(self, lamb = 1e3):
        super().__init__()
        self.lamb = nn.Parameter(lamb * torch.ones(1))
        #self.inv_lamb = nn.Parameter((1./lamb) * torch.ones(1))

    def forward(self, x):
        B, D, H, W = x.shape
        N = H * W
        identity = torch.eye(N).cuda()
        # reshape x, s.t. we can use the gmp formulation as a global pooling operation
        x = x.view(B, D, N)
        x = x.permute(0, 2, 1)
        # compute the linear kernel
        K = torch.bmm(x, x.permute(0, 2, 1))
        # solve the linear system (K + lambda * I) * alpha = ones
        A = K + self.lamb * identity
        o = torch.ones(B, N, 1).cuda()
        #alphas, _ = torch.gesv(o, A) # tested using pytorch 1.0.1
        alphas = torch.linalg.solve(A,o) # TODO check it again
        alphas = alphas.view(B, 1, -1)        
        xi = torch.bmm(alphas, x)
        xi = xi.view(B, -1)
        return xi

class OTKPool(nn.Module):
    """ Pooling with Optimal Transport Kernel Embedding
    """
    def __init__(self, in_dim=512, out_size=1, heads=1, eps=25, max_iter=1):
        super().__init__()
        self.otk = OTKernel(in_dim=in_dim, out_size=out_size, heads=heads, eps=eps, max_iter=max_iter)

    def forward(self, x):
        x = prepare_input(x)
        x = self.otk(x)
        return x

class GEPooling(nn.Module):
    def __init__(self, channels=512, feat_size=None, extra_params=False, extent=0, output_size=1):
        super(GEPooling, self).__init__()

        self.ge = GatherExcite(channels=channels, feat_size=feat_size, extra_params=extra_params, extent=extent)

        self.relu = nn.ReLU(inplace=True)
        self.avgpool = nn.AdaptiveAvgPool2d(output_size)

    def forward(self, x):
        residual = x
        x = self.ge(x)
        x += residual
        x = self.relu(x)
        x = self.avgpool(x)
        return x

class HOWPooling(nn.Module):
    def __init__(self, input_dim = 512, dim_reduction = 128, kernel_size = 3):
        super(HOWPooling, self).__init__()
        self.kernel_size = kernel_size
        self.dimreduction = ConvDimReduction(input_dim, dim_reduction)

    def L2Attention(self, x):
        return (x.pow(2.0).sum(1) + 1e-10).sqrt().squeeze(0)

    def smoothing_avg_pooling(self, feats):
        """Smoothing average pooling
        :param torch.Tensor feats: Feature map
        :param int kernel_size: kernel size of pooling
        :return torch.Tensor: Smoothend feature map
        """
        pad = self.kernel_size // 2
        return F.avg_pool2d(feats, (self.kernel_size, self.kernel_size), stride=1, padding=pad,
                            count_include_pad=False)        

    def l2n(self, x, eps=1e-6):
        return x / (torch.norm(x, p=2, dim=1, keepdim=True) + eps).expand_as(x)

    def forward(self, x):

        weights = self.L2Attention(x)
        x = self.smoothing_avg_pooling(x)
        x = self.dimreduction(x)
        x = (x * weights.unsqueeze(1)).sum((-2, -1))
        return self.l2n(x)

class ConvDimReduction(nn.Conv2d):
    """Dimensionality reduction as a convolutional layer
    :param int input_dim: Network out_channels
    :param in dim: Whitening out_channels, for dimensionality reduction
    """

    def __init__(self, input_dim, dim):
        super().__init__(input_dim, dim, (1, 1), padding=0, bias=True)

    def pcawhitenlearn_shrinkage(X, s=1.0):
        """Learn PCA whitening with shrinkage from given descriptors"""
        N = X.shape[0]

        # Learning PCA w/o annotations
        m = X.mean(axis=0, keepdims=True)
        Xc = X - m
        Xcov = np.dot(Xc.T, Xc)
        Xcov = (Xcov + Xcov.T) / (2*N)
        eigval, eigvec = np.linalg.eig(Xcov)
        order = eigval.argsort()[::-1]
        eigval = eigval[order]
        eigvec = eigvec[:, order]

        eigval = np.clip(eigval, a_min=1e-14, a_max=None)
        P = np.dot(np.linalg.inv(np.diag(np.power(eigval, 0.5*s))), eigvec.T)

        return m, P.T

    def initialize_pca_whitening(self, des):
        """Initialize PCA whitening from given descriptors. Return tuple of shift and projection."""
        m, P = self.pcawhitenlearn_shrinkage(des)
        m, P = m.T, P.T

        projection = torch.Tensor(P[:self.weight.shape[0], :]).unsqueeze(-1).unsqueeze(-1)
        self.weight.data = projection.to(self.weight.device)

        projected_shift = -torch.mm(torch.FloatTensor(P), torch.FloatTensor(m)).squeeze()
        self.bias.data = projected_shift[:self.weight.shape[0]].to(self.bias.device)
        return m.T, P.T

class ScaledDotProductAttention(nn.Module):
    """
    Scaled Dot-Product Attention proposed in "Attention Is All You Need"
    Compute the dot products of the query with all keys, divide each by sqrt(dim),
    and apply a softmax function to obtain the weights on the values
    Args: dim, mask
        dim (int): dimention of attention
        mask (torch.Tensor): tensor containing indices to be masked
    Inputs: query, key, value, mask
        - **query** (batch, q_len, d_model): tensor containing projection vector for decoder.
        - **key** (batch, k_len, d_model): tensor containing projection vector for encoder.
        - **value** (batch, v_len, d_model): tensor containing features of the encoded input sequence.
        - **mask** (-): tensor containing indices to be masked
    Returns: context, attn
        - **context**: tensor containing the context vector from attention mechanism.
        - **attn**: tensor containing the attention (alignment) from the encoder outputs.
    """
    def __init__(self, dim: int):
        super(ScaledDotProductAttention, self).__init__()
        self.sqrt_dim = np.sqrt(dim)

    def forward(self, query: Tensor, key: Tensor, value: Tensor, mask: Optional[Tensor] = None) -> Tuple[Tensor, Tensor]:
        score = torch.bmm(query, key.transpose(1, 2)) / self.sqrt_dim

        if mask is not None:
            score.masked_fill_(mask.view(score.size()), -float('Inf'))

        attn = F.softmax(score, -1)
        context = torch.bmm(attn, value)
        return context, attn

class ViTPooling(nn.Module):
    """
    Multi-Head Attention proposed in "Attention Is All You Need"
    Instead of performing a single attention function with d_model-dimensional keys, values, and queries,
    project the queries, keys and values h times with different, learned linear projections to d_head dimensions.
    These are concatenated and once again projected, resulting in the final values.
    Multi-head attention allows the model to jointly attend to information from different representation
    subspaces at different positions.
    MultiHead(Q, K, V) = Concat(head_1, ..., head_h) · W_o
        where head_i = Attention(Q · W_q, K · W_k, V · W_v)
    Args:
        d_model (int): The dimension of keys / values / quries (default: 512)
        num_heads (int): The number of attention heads. (default: 8)
    Inputs: query, key, value, mask
        - **query** (batch, q_len, d_model): In transformer, three different ways:
            Case 1: come from previoys decoder layer
            Case 2: come from the input embedding
            Case 3: come from the output embedding (masked)
        - **key** (batch, k_len, d_model): In transformer, three different ways:
            Case 1: come from the output of the encoder
            Case 2: come from the input embeddings
            Case 3: come from the output embedding (masked)
        - **value** (batch, v_len, d_model): In transformer, three different ways:
            Case 1: come from the output of the encoder
            Case 2: come from the input embeddings
            Case 3: come from the output embedding (masked)
        - **mask** (-): tensor containing indices to be masked
    Returns: output, attn
        - **output** (batch, output_len, dimensions): tensor containing the attended output features.
        - **attn** (batch * num_heads, v_len): tensor containing the attention (alignment) from the encoder outputs.
    """
    def __init__(self, d_model: int = 512, num_heads: int = 8):
        super(ViTPooling, self).__init__()

        assert d_model % num_heads == 0, "d_model % num_heads should be zero."

        self.d_head = int(d_model / num_heads)
        self.num_heads = num_heads
        self.scaled_dot_attn = ScaledDotProductAttention(self.d_head)
        self.query_proj = nn.Linear(d_model, self.d_head * num_heads)
        self.key_proj = nn.Linear(d_model, self.d_head * num_heads)
        self.value_proj = nn.Linear(d_model, self.d_head * num_heads)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))

    def forward(
            self,
            x: Tensor,
            mask: Optional[Tensor] = None
    ) -> Tuple[Tensor, Tensor]:

        x = prepare_input(x)
        cls_token = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_token, x), dim=1)

        query = self.query_proj(x).view(B, -1, self.num_heads, self.d_head)  # BxQ_LENxNxD
        key = self.key_proj(x).view(B, -1, self.num_heads, self.d_head)      # BxK_LENxNxD
        value = self.value_proj(x).view(B, -1, self.num_heads, self.d_head)  # BxV_LENxNxD

        query = query.permute(2, 0, 1, 3).contiguous().view(B * self.num_heads, -1, self.d_head)  # BNxQ_LENxD
        key = key.permute(2, 0, 1, 3).contiguous().view(B * self.num_heads, -1, self.d_head)      # BNxK_LENxD
        value = value.permute(2, 0, 1, 3).contiguous().view(B * self.num_heads, -1, self.d_head)  # BNxV_LENxD

        if mask is not None:
            mask = mask.unsqueeze(1).repeat(1, self.num_heads, 1, 1)  # BxNxQ_LENxK_LEN

        context, attn = self.scaled_dot_attn(query, key, value, mask)

        context = context.view(self.num_heads, B, -1, self.d_head)
        context = context.permute(1, 2, 0, 3).contiguous().view(B, -1, self.num_heads * self.d_head)  # BxTxND

        return context[:, 0]

class SlotPooling(nn.Module):
    def __init__(self, num_slots, dim, iters = 3, eps = 1e-8, hidden_dim = 128):
        super().__init__()
        self.num_slots = num_slots
        self.iters = iters
        self.eps = eps
        self.scale = dim ** -0.5

        self.slots_mu = nn.Parameter(torch.randn(1, 1, dim))

        self.slots_logsigma = nn.Parameter(torch.zeros(1, 1, dim))
        init.xavier_uniform_(self.slots_logsigma)

        self.to_q = nn.Linear(dim, dim)
        self.to_k = nn.Linear(dim, dim)
        self.to_v = nn.Linear(dim, dim)

        self.gru = nn.GRUCell(dim, dim)

        hidden_dim = max(dim, hidden_dim)

        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.ReLU(inplace = True),
            nn.Linear(hidden_dim, dim)
        )

        self.norm_input  = nn.LayerNorm(dim)
        self.norm_slots  = nn.LayerNorm(dim)
        self.norm_pre_ff = nn.LayerNorm(dim)

    def forward(self, inputs, num_slots = None):
        inputs = prepare_input(inputs)
        b, n, d, device, dtype = *inputs.shape, inputs.device, inputs.dtype
        n_s = num_slots if num_slots is not None else self.num_slots

        mu = self.slots_mu.expand(b, n_s, -1)
        sigma = self.slots_logsigma.exp().expand(b, n_s, -1)

        slots = mu + sigma * torch.randn(mu.shape, device = device, dtype = dtype)

        inputs = self.norm_input(inputs)        
        k, v = self.to_k(inputs), self.to_v(inputs)

        for _ in range(self.iters):
            slots_prev = slots

            slots = self.norm_slots(slots)
            q = self.to_q(slots)

            dots = torch.einsum('bid,bjd->bij', q, k) * self.scale
            attn = dots.softmax(dim=1) + self.eps

            attn = attn / attn.sum(dim=-1, keepdim=True)

            updates = torch.einsum('bjd,bij->bid', v, attn)

            slots = self.gru(
                updates.reshape(-1, d),
                slots_prev.reshape(-1, d)
            )

            slots = slots.reshape(b, -1, d)
            slots = slots + self.mlp(self.norm_pre_ff(slots))
        slots = slots.max(dim=1)[0]

        return slots

class CAPooling(nn.Module):
    def __init__(self, embed_dim = 512, 
                 num_heads=4, 
                 iterations=1, 
                 qkv_bias=True, 
                 norm_layer = partial(nn.LayerNorm, eps=1e-6), 
                 act_layer=nn.GELU,
                 qk_scale=None,
                 init_scale=1e-5,
                 mlp_ratio_clstk = 4.0):
        super(CAPooling, self).__init__()

        from timm.layers import Mlp
        from timm.layers import trunc_normal_

        self.depth_token_only = iterations

        self.blocks_token_only = nn.ModuleList([
            LayerScale_Block_CA(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio_clstk, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=0.0, attn_drop=0.0, drop_path=0.0, norm_layer=norm_layer,
                act_layer=act_layer,Attention_block=Class_Attention,
                Mlp_block=Mlp,init_values=init_scale)
            for i in range(self.depth_token_only)])

        self.norm = nn.LayerNorm(embed_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        trunc_normal_(self.cls_token, std=.02)

    def forward(self, x):
        x = prepare_input(x)
        cls_tokens = self.cls_token.expand(B, -1, -1)  

        for i , blk in enumerate(self.blocks_token_only):
            cls_tokens = blk(x,cls_tokens)

        x = torch.cat((cls_tokens, x), dim=1)

        x = self.norm(x)
        return x[:, 0]



class Class_Attention(nn.Module):
    # taken from https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
    # with slight modifications to do CA 
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.k = nn.Linear(dim, dim, bias=qkv_bias)
        self.v = nn.Linear(dim, dim, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):

        x = prepare_input(x)
        B, N, C = x.shape
        q = self.q(x[:,0]).unsqueeze(1).reshape(B, 1, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        k = self.k(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        q = q * self.scale
        v = self.v(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        attn = (q @ k.transpose(-2, -1)) 
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x_cls = (attn @ v).transpose(1, 2).reshape(B, 1, C)
        x_cls = self.proj(x_cls)
        x_cls = self.proj_drop(x_cls)

        return x_cls     


class LayerScale_Block_CA(nn.Module):
    # taken from https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
    # with slight modifications to add CA and LayerScale
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, Attention_block = Class_Attention,
                 Mlp_block=None,init_values=1e-4):
        super().__init__()

        from timm.layers import DropPath

        self.norm1 = norm_layer(dim)
        self.attn = Attention_block(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp_block(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.gamma_1 = nn.Parameter(init_values * torch.ones((dim)),requires_grad=True)
        self.gamma_2 = nn.Parameter(init_values * torch.ones((dim)),requires_grad=True)

    def forward(self, x, x_cls):

        u = torch.cat((x_cls,x),dim=1)
        x_cls = x_cls + self.drop_path(self.gamma_1 * self.attn(self.norm1(u)))
        x_cls = x_cls + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x_cls)))
        
        return x_cls 