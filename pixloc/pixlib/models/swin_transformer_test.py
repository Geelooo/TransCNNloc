""" Swin Transformer
A PyTorch impl of : `Swin Transformer: Hierarchical Vision Transformer using Shifted Windows`
    - https://arxiv.org/pdf/2103.14030

Code/weights from https://github.com/microsoft/Swin-Transformer

"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
import numpy as np
import copy
from einops import rearrange
from typing import Optional


def drop_path_f(x, drop_prob: float = 0., training: bool = False):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).

    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.

    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    # work with diff dim tensors, not just 2D ConvNets
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = keep_prob + \
        torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """

    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path_f(x, self.drop_prob, self.training)

# 正向操作


def window_partition(x, window_size: int):
    """
    将feature map按照window_size划分成一个个没有重叠的window
    Args:
        x: (B, H, W, C)
        window_size (int): window size(M)

    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size,
               W // window_size, window_size, C)
    # permute: [B, H//Mh, Mh, W//Mw, Mw, C] -> [B, H//Mh, W//Mh, Mw, Mw, C] 并变成内存连续的数据
    # view: [B, H//Mh, W//Mw, Mh, Mw, C] -> [B*num_windows, Mh, Mw, C]  H//Mh* W//Mw 为window的个数
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous(
    ).view(-1, window_size, window_size, C)
    return windows

# 反向操作


def window_reverse(windows, window_size: int, H: int, W: int):  # 分割之前的H和W
    """
    将一个个window还原成一个feature map
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int): Window size(M)
        H (int): Height of image
        W (int): Width of image

    Returns:
        x: (B, H, W, C)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    # view: [B*num_windows, Mh, Mw, C] -> [B, H//Mh, W//Mw, Mh, Mw, C]
    x = windows.view(B, H // window_size, W // window_size,
                     window_size, window_size, -1)
    # permute: [B, H//Mh, W//Mw, Mh, Mw, C] -> [B, H//Mh, Mh, W//Mw, Mw, C]
    # view: [B, H//Mh, Mh, W//Mw, Mw, C] -> [B, H, W, C]
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


class PatchEmbed(nn.Module):
    """
    2D Image to Patch Embedding
    """

    def __init__(self, patch_size=4, in_c=3, embed_dim=96, norm_layer=None):
        super().__init__()
        patch_size = (patch_size, patch_size)
        self.patch_size = patch_size
        self.in_chans = in_c
        self.embed_dim = embed_dim
        self.proj = nn.Conv2d(
            in_c, embed_dim, kernel_size=patch_size, stride=patch_size)  # 下采样操作通过卷积实现
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        _, _, H, W = x.shape  # 样本数*通道数*高度*宽度

        # padding
        # 如果输入图片的H，W不是patch_size的整数倍，需要进行padding
        pad_input = (H % self.patch_size[0] != 0) or (
            W % self.patch_size[1] != 0)
        if pad_input:
            # to pad the last 3 dimensions,
            # (W_left, W_right, H_top,H_bottom, C_front, C_back) 如果需要padding，是在右侧和底侧进行填充
            x = F.pad(x, (0, self.patch_size[1] - W % self.patch_size[1],
                          0, self.patch_size[0] - H % self.patch_size[0],
                          0, 0))

        # 下采样patch_size倍
        x = self.proj(x)
        _, _, H, W = x.shape
        # flatten: [B, C, H, W] -> [B, C, HW]
        # transpose: [B, C, HW] -> [B, HW, C]
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)  # 如果normlayer为空，则输入=输出
        return x, H, W


class PatchMerging(nn.Module):
    r""" Patch Merging Layer.

    Args:
        dim (int): Number of input channels.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.norm = norm_layer(4 * dim)
        self.reduction = nn.Linear(
            4 * dim, 2 * dim, bias=False)  # 最后的linear层 c减半

    def forward(self, x, H, W):
        """
        x: B, H*W, C
        """
        B, L, C = x.shape
        assert L == H * W, "PatchMerging error"

        x = x.view(B, H, W, C)

        # padding
        # 如果输入feature map的H，W不是2的整数倍，需要进行padding
        pad_input = (H % 2 == 1) or (W % 2 == 1)
        if pad_input:
            # to pad the last 3 dimensions, starting from the last dimension and moving forward.
            # (C_front, C_back, W_left, W_right, H_top, H_bottom)
            # 注意这里的Tensor通道是[B, H, W, C]，所以会和官方文档有些不同
            x = F.pad(x, (0, 0, 0, W % 2, 0, H % 2))

        x0 = x[:, 0::2, 0::2, :]  # [B, H/2, W/2, C]  按步长抽样
        x1 = x[:, 1::2, 0::2, :]  # [B, H/2, W/2, C]
        x2 = x[:, 0::2, 1::2, :]  # [B, H/2, W/2, C]
        x3 = x[:, 1::2, 1::2, :]  # [B, H/2, W/2, C]
        x = torch.cat([x0, x1, x2, x3], -1)  # [B, H/2, W/2, 4*C]
        x = x.view(B, -1, 4 * C)  # [B, H/2*W/2, 4*C]

        x = self.norm(x)
        x = self.reduction(x)  # [B, H/2*W/2, 2*C]

        return x


class PatchExpand(nn.Module):
    def __init__(self, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.expand = nn.Linear(dim, 2*dim, bias=False)
        self.norm = norm_layer(dim // 2)

    def forward(self, x, H, W):
        """
        x: B, H*W, C
        """
        x = self.expand(x)
        B, L, C = x.shape
        assert L == H * W, "PatchExpand error"

        x = x.view(B, H, W, C)
        x = rearrange(x, 'b h w (p1 p2 c)-> b (h p1) (w p2) c',
                      p1=2, p2=2, c=C//4)
        x = x.view(B, -1, C//4)
        x = self.norm(x)
        return x, H*2, W*2


class Patch_X4(nn.Module):
    def __init__(self, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.linear = nn.Linear(dim, dim*16, bias=False)
        self.norm = norm_layer(dim)
        self.dim = dim

    def forward(self, x, H, W):
        x = self.linear(x)
        B, L, C = x.shape
        assert L == H*W, "feature_X4 error"

        x = x.view(B, H, W, C)
        x = rearrange(x, 'b h w (p1 p2 c)-> b (h p1) (w p2) c',
                      p1=4, p2=4, c=C//16)
        x = x.view(B, -1, self.dim)
        x = self.norm(x)
        return x, H*4, W*4


class Mlp(nn.Module):
    """ MLP as used in Vision Transformer, MLP-Mixer and related networks
    """

    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop)
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop2 = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x


class WindowAttention(nn.Module):
    r""" Window based multi-head self attention (W-MSA) module with relative position bias.
    It supports both of shifted and non-shifted window.

    Args:
        dim (int): Number of input channels.
        window_size (tuple[int]): The height and width of the window.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
    """

    def __init__(self, dim, window_size, num_heads, qkv_bias=True, attn_drop=0., proj_drop=0.):

        super().__init__()
        self.dim = dim
        self.window_size = window_size  # [Mh, Mw]
        self.num_heads = num_heads
        head_dim = dim // num_heads  # 每一个head对应的dim
        self.scale = head_dim ** -0.5   #

        # define a parameter table of relative position bias
        # 每一个head都有一个table
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))  # [2*Mh-1 * 2*Mw-1, nH] num_heads

        # get pair-wise relative position index for each token inside the window
        # 生成相对位置索引
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid(
            [coords_h, coords_w]))  # [2, Mh, Mw]
        coords_flatten = torch.flatten(coords, 1)  # [2, Mh*Mw]  上行下列
        # [2, Mh*Mw, 1] - [2, 1, Mh*Mw]  广播复制相减
        relative_coords = coords_flatten[:, :, None] - \
            coords_flatten[:, None, :]  # [2, Mh*Mw, Mh*Mw]
        relative_coords = relative_coords.permute(
            1, 2, 0).contiguous()  # [Mh*Mw, Mh*Mw, 2]
        relative_coords[:, :, 0] += self.window_size[0] - \
            1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)  # [Mh*Mw, Mh*Mw]
        self.register_buffer("relative_position_index",
                             relative_position_index)  # 将相对位置索引放入模型缓存当中

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        nn.init.trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask: Optional[torch.Tensor] = None):
        """
        Args:
            x: input features with shape of (num_windows*B, Mh*Mw, C)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """
        # [batch_size*num_windows, Mh*Mw, total_embed_dim]
        B_, N, C = x.shape
        # qkv(): -> [batch_size*num_windows, Mh*Mw, 3 * total_embed_dim]
        # reshape: -> [batch_size*num_windows, Mh*Mw, 3, num_heads, embed_dim_per_head] 每一个head对应的qkv 维度
        # permute: -> [3, batch_size*num_windows, num_heads, Mh*Mw, embed_dim_per_head]
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C //
                                  self.num_heads).permute(2, 0, 3, 1, 4)
        # [batch_size*num_windows, num_heads, Mh*Mw, embed_dim_per_head]
        # make torchscript happy (cannot use tensor as tuple)
        q, k, v = qkv.unbind(0)

        # transpose: -> [batch_size*num_windows, num_heads, embed_dim_per_head, Mh*Mw]
        # @: multiply -> [batch_size*num_windows, num_heads, Mh*Mw, Mh*Mw]
        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        # relative_position_bias_table.view: [Mh*Mw*Mh*Mw,nH] -> [Mh*Mw,Mh*Mw,nH]
        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)
        relative_position_bias = relative_position_bias.permute(
            2, 0, 1).contiguous()  # [nH, Mh*Mw, Mh*Mw]
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            # mask: [nW, Mh*Mw, Mh*Mw]
            nW = mask.shape[0]  # num_windows
            # attn.view: [batch_size, num_windows, num_heads, Mh*Mw, Mh*Mw]
            # mask.unsqueeze: [1, nW, 1, Mh*Mw, Mh*Mw]
            attn = attn.view(B_ // nW, nW, self.num_heads, N,
                             N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        # @: multiply -> [batch_size*num_windows, num_heads, Mh*Mw, embed_dim_per_head]
        # transpose: -> [batch_size*num_windows, Mh*Mw, num_heads, embed_dim_per_head]
        # reshape: -> [batch_size*num_windows, Mh*Mw, total_embed_dim]
        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class SwinTransformerBlock(nn.Module):
    r""" Swin Transformer Block.

    Args:
        dim (int): Number of input channels.
        num_heads (int): Number of attention heads.
        window_size (int): Window size.
        shift_size (int): Shift size for SW-MSA.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, dim, num_heads, window_size=7, shift_size=0,
                 mlp_ratio=4., qkv_bias=True, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        assert 0 <= self.shift_size < self.window_size, "shift_size must in 0->window_size"

        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(
            dim, window_size=(
                self.window_size, self.window_size), num_heads=num_heads, qkv_bias=qkv_bias,
            attn_drop=attn_drop, proj_drop=drop)

        self.drop_path = DropPath(
            drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim,
                       act_layer=act_layer, drop=drop)

    def forward(self, x, attn_mask):
        H, W = self.H, self.W
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)

        # pad feature maps to multiples of window size
        # 把feature map给pad到window size的整数倍
        pad_l = pad_t = 0
        pad_r = (self.window_size - W % self.window_size) % self.window_size
        pad_b = (self.window_size - H % self.window_size) % self.window_size
        x = F.pad(x, (0, 0, pad_l, pad_r, pad_t, pad_b))
        _, Hp, Wp, _ = x.shape

        # cyclic shift
        if self.shift_size > 0:  # swmsa  需要发生平移
            shifted_x = torch.roll(
                x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x
            attn_mask = None

        # partition windows
        x_windows = window_partition(
            shifted_x, self.window_size)  # [nW*B, Mh, Mw, C]
        # [nW*B, Mh*Mw, C]
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)

        # W-MSA/SW-MSA
        attn_windows = self.attn(x_windows, mask=attn_mask)  # [nW*B, Mh*Mw, C]

        # merge windows
        # [nW*B, Mh, Mw, C]
        attn_windows = attn_windows.view(-1,
                                         self.window_size, self.window_size, C)
        shifted_x = window_reverse(
            attn_windows, self.window_size, Hp, Wp)  # [B, H', W', C]

        # reverse cyclic shift
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(
                self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x

        if pad_r > 0 or pad_b > 0:
            # 把前面pad的数据移除掉
            x = x[:, :H, :W, :].contiguous()

        x = x.view(B, H * W, C)

        # FFN
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x


class BasicLayer(nn.Module):
    """
    A basic Swin Transformer layer for one stage.

    Args:
        dim (int): Number of input channels.
        depth (int): Number of blocks.
        num_heads (int): Number of attention heads.
        window_size (int): Local window size.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
    """

    def __init__(self, dim, depth, num_heads, window_size,
                 mlp_ratio=4., qkv_bias=True, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, downsample=None, use_checkpoint=False):
        super().__init__()
        self.dim = dim
        self.depth = depth
        self.window_size = window_size
        self.use_checkpoint = use_checkpoint
        self.shift_size = window_size // 2    # 窗口大小/2 向下取整

        # build blocks
        self.blocks = nn.ModuleList([
            SwinTransformerBlock(
                dim=dim,
                num_heads=num_heads,
                window_size=window_size,
                # wmsa和swmsa成对按序出现
                shift_size=0 if (i % 2 == 0) else self.shift_size,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                drop=drop,
                attn_drop=attn_drop,
                drop_path=drop_path[i] if isinstance(
                    drop_path, list) else drop_path,
                norm_layer=norm_layer)
            for i in range(depth)])

        # patch merging layer
        # 如果不是最后一个stage则实例化一个pm层，如果是就打成None
        if downsample is not None:
            self.downsample = downsample(dim=dim, norm_layer=norm_layer)
        else:
            self.downsample = None

    def create_mask(self, x, H, W):
        # calculate attention mask for SW-MSA
        # 向上取整以后再乘以windowsize保证是整数倍
        # 保证Hp和Wp是window_size的整数倍
        Hp = int(np.ceil(H / self.window_size)) * self.window_size
        Wp = int(np.ceil(W / self.window_size)) * self.window_size
        # 拥有和feature map一样的通道排列顺序，方便后续window_partition
        img_mask = torch.zeros(
            (1, Hp, Wp, 1), device=x.device)  # [1, Hp, Wp, 1]
        h_slices = (slice(0, -self.window_size),
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None))
        w_slices = (slice(0, -self.window_size),
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None))
        cnt = 0
        for h in h_slices:
            for w in w_slices:
                img_mask[:, h, w, :] = cnt
                cnt += 1
                # num_win 窗高 窗宽 1
        mask_windows = window_partition(
            img_mask, self.window_size)  # [nW, Mh, Mw, 1]
        # [nW, Mh*Mw]
        mask_windows = mask_windows.view(-1,
                                         self.window_size * self.window_size)
        # [nW, 1, Mh*Mw] - [nW, Mh*Mw, 1]
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        # [nW, Mh*Mw, Mh*Mw]
        attn_mask = attn_mask.masked_fill(
            attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        return attn_mask

    def forward(self, x, H, W):
        # [nW, Mh*Mw, Mh*Mw]  SWMSA时所用的蒙板，接收HW解决输入
        attn_mask = self.create_mask(x, H, W)
        for blk in self.blocks:                # 图片多尺度的问题
            blk.H, blk.W = H, W
            if not torch.jit.is_scripting() and self.use_checkpoint:  # 默认这个分支不会进去的
                x = checkpoint.checkpoint(blk, x, attn_mask)
            else:
                x = blk(x, attn_mask)
        if self.downsample is not None:
            x = self.downsample(x, H, W)
            H, W = (H + 1) // 2, (W + 1) // 2  # 偶数的话不变

        return x, H, W


class BasicLayer_up(nn.Module):
    """ A basic Swin Transformer layer for one stage.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resolution.
        depth (int): Number of blocks.
        num_heads (int): Number of attention heads.
        window_size (int): Local window size.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
    """

    def __init__(self, dim, depth, num_heads, window_size,
                 mlp_ratio=4., qkv_bias=True, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, upsample=None, use_checkpoint=False):

        super().__init__()
        self.dim = dim
        self.depth = depth
        self.window_size = window_size
        self.use_checkpoint = use_checkpoint
        self.shift_size = window_size // 2

        # build blocks
        self.blocks = nn.ModuleList([
            SwinTransformerBlock(dim=dim,
                                 num_heads=num_heads,
                                 window_size=window_size,
                                 shift_size=0 if (
                                     i % 2 == 0) else self.shift_size,
                                 mlp_ratio=mlp_ratio,
                                 qkv_bias=qkv_bias,
                                 drop=drop,
                                 attn_drop=attn_drop,
                                 drop_path=drop_path[i] if isinstance(
                                     drop_path, list) else drop_path,
                                 norm_layer=norm_layer)
            for i in range(depth)])

        # patch merging layer
        if upsample is not None:
            self.upsample = PatchExpand(dim=dim, norm_layer=norm_layer)
        else:
            self.upsample = None

    def create_mask(self, x, H, W):
        # calculate attention mask for SW-MSA
        # 向上取整以后再乘以windowsize保证是整数倍
        # 保证Hp和Wp是window_size的整数倍
        Hp = int(np.ceil(H / self.window_size)) * self.window_size
        Wp = int(np.ceil(W / self.window_size)) * self.window_size
        # 拥有和feature map一样的通道排列顺序，方便后续window_partition
        img_mask = torch.zeros(
            (1, Hp, Wp, 1), device=x.device)  # [1, Hp, Wp, 1]
        h_slices = (slice(0, -self.window_size),
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None))
        w_slices = (slice(0, -self.window_size),
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None))
        cnt = 0
        for h in h_slices:
            for w in w_slices:
                img_mask[:, h, w, :] = cnt
                cnt += 1
                # num_win 窗高 窗宽 1
        mask_windows = window_partition(
            img_mask, self.window_size)  # [nW, Mh, Mw, 1]
        # [nW, Mh*Mw]
        mask_windows = mask_windows.view(-1,
                                         self.window_size * self.window_size)
        # [nW, 1, Mh*Mw] - [nW, Mh*Mw, 1]
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        # [nW, Mh*Mw, Mh*Mw]
        attn_mask = attn_mask.masked_fill(
            attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        return attn_mask

    def forward(self, x, H, W):
        # [nW, Mh*Mw, Mh*Mw]  SWMSA时所用的蒙板，接收HW解决输入
        attn_mask = self.create_mask(x, H, W)
        for blk in self.blocks:                # 图片多尺度的问题
            blk.H, blk.W = H, W
            if not torch.jit.is_scripting() and self.use_checkpoint:  # 默认这个分支不会进去的
                x = checkpoint.checkpoint(blk, x, attn_mask)
            else:
                x = blk(x, attn_mask)
        if self.upsample is not None:
            x, H, W = self.upsample(x, H, W)

        return x, H, W


class SwinUnet(nn.Module):
    r""" Swin Transformer
        A PyTorch impl of : `Swin Transformer: Hierarchical Vision Transformer using Shifted Windows`  -
          https://arxiv.org/pdf/2103.14030

    Args:
        patch_size (int | tuple(int)): Patch size. Default: 4   下采样次数
        in_chans (int): Number of input image channels. Default: 3  图片输入深度
        num_classes (int): Number of classes for classification head. Default: 1000 分类类别数
        embed_dim (int): Patch embedding dimension. Default: 96  映射得到的维度数C 
        depths (tuple(int)): Depth of each Swin Transformer layer.  每一个stage重复使用transformer次数2 2 6 2 
        num_heads (tuple(int)): Number of attention heads in different layers.   多头注意力机制中的头数3 6 12 24
        window_size (int): Window size. Default: 7   WMSA和SWMSA采用的窗口大小
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4  mlp全连接层翻的倍数
        qkv_bias (bool): If True, add a learnable bias to query, key, value. Default: True   是否使用偏置
        drop_rate (float): Dropout rate. Default: 0   在pos_drop中用到，以及mlp和其他地方
        attn_drop_rate (float): Attention dropout rate. Default: 0  多头注意力机制中使用的drop rate
        drop_path_rate (float): Stochastic depth rate. Default: 0.1 swin-block中
        norm_layer (nn.Module): Normalization layer. Default: nn.LayerNorm.
        patch_norm (bool): If True, add normalization after patch embedding. Default: True
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False 使用能节省内存
    """
    def __init__(self):
        super().__init__()

        self.patch_size=4
        self.in_chans=3
        self.embed_dim=96
        self.depths=(2, 2, 6, 2)
        self.num_heads=(3, 6, 12, 24)
        self.window_size=7
        self.mlp_ratio=4.
        self.qkv_bias=True
        self.drop_rate=0.
        self.attn_drop_rate=0.
        self.drop_path_rate=0.1
        self.norm_layer=nn.LayerNorm
        self.patch_norm=True
        self.use_checkpoint=False

        self.num_layers = len(self.depths)  # 4
        # stage4输出特征矩阵的channels
        self.num_features = int(self.embed_dim * 2 ** (self.num_layers - 1))  # 96*8

        # split image into non-overlapping patches
        # 对应着图中的patch partition和linear embedding
        self.patch_embed = PatchEmbed(
            patch_size=self.patch_size, in_c=self.in_chans, embed_dim=self.embed_dim,
            norm_layer=self.norm_layer if self.patch_norm else None)
        self.pos_drop = nn.Dropout(p=self.drop_rate)

        # stochastic depth  从0->逐渐放大到提供的值
        # 利用自带得线性系统，定义初始和最终大小，以及步长，所以每个st-blk都为不一样的rate
        dpr = [x.item() for x in torch.linspace(0, self.drop_path_rate,
                                                sum(self.depths))]  # stochastic depth decay rule

        # build encoder
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):  # 0->3
            # 注意这里构建的stage和论文图中有些差异
            # 这里的stage不包含该stage的patch_merging层，包含的是下个stage的
            layers = BasicLayer(dim=int(self.embed_dim * 2 ** i_layer),  # 96 96*2 ..每次乘
                                depth=self.depths[i_layer],
                                num_heads=self.num_heads[i_layer],
                                window_size=self.window_size,
                                mlp_ratio=self.mlp_ratio,
                                qkv_bias=self.qkv_bias,
                                drop=self.drop_rate,
                                attn_drop=self.attn_drop_rate,
                                # 当前的stage中每一个st-blk所采用的drop_path_rate
                                drop_path=dpr[sum(self.depths[:i_layer]):sum(
                                    self.depths[:i_layer + 1])],
                                norm_layer=self.norm_layer,
                                # 4-1=3 构建前三个stage的时候是有patch merging
                                downsample=PatchMerging if (
                                    i_layer < self.num_layers - 1) else None,
                                use_checkpoint=self.use_checkpoint)
            self.layers.append(layers)

        self.layers_up = nn.ModuleList()
        self.concat_back = nn.ModuleList()
        # build decoder
        for i_layer in range(self.num_layers):  # 0->3
            concat_linear = nn.Linear(2*int(self.embed_dim*2**(self.num_layers-1-i_layer)), int(
                self.embed_dim*2**(self.num_layers-1-i_layer)))if i_layer > 0 else nn.Identity()
            if i_layer == 0:
                layer_up = PatchExpand(
                    int(self.embed_dim*2**(self.num_layers-1-i_layer)))
            else:
                layer_up = BasicLayer_up(dim=int(self.embed_dim*2**(self.num_layers-1-i_layer)),
                                         depth=self.depths[(
                                             self.num_layers-1-i_layer)],
                                         num_heads=self.num_heads[(
                                             self.num_layers-1-i_layer)],
                                         window_size=self.window_size,
                                         mlp_ratio=self.mlp_ratio,
                                         qkv_bias=self.qkv_bias,
                                         drop=self.drop_rate, attn_drop=self.attn_drop_rate,
                                         drop_path=dpr[sum(self.depths[:(
                                             self.num_layers-1-i_layer)]):sum(self.depths[:(self.num_layers-1-i_layer) + 1])],
                                         norm_layer=self.norm_layer,
                                         upsample=PatchExpand if (
                    i_layer < self.num_layers - 1) else None,
                    use_checkpoint=self.use_checkpoint)
            self.layers_up.append(layer_up)
            self.concat_back.append(concat_linear)

        self.norm = self.norm_layer(self.num_features)
        self.norm_up = self.norm_layer(self.embed_dim)

        # X4层
        self.feature_X4 = Patch_X4(self.embed_dim)

        # 概率图的预测
        self.utc0 = nn.Conv2d(
            in_channels=self.embed_dim*4, out_channels=1, kernel_size=1, bias=True)
        self.utc1 = nn.Conv2d(
            in_channels=self.embed_dim*2, out_channels=1, kernel_size=1, bias=True)
        self.utc2 = nn.Conv2d(
            in_channels=self.embed_dim, out_channels=1, kernel_size=1, bias=True)
        self.utc3 = nn.Conv2d(
            in_channels=self.embed_dim, out_channels=1, kernel_size=1, bias=True)

        # 初始化权重
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):

        # encoder forward
        # x: [B, L, C]
        B = x.shape[0]
        x, H, W = self.patch_embed(x)
        x = self.pos_drop(x)
        skip_connection = []
        for layer in self.layers:
            skip_connection.append(x)
            x, H, W = layer(x, H, W)

        x = self.norm(x)  # [B, L, C]

        features = []
        # decoder backward
        for idx, layer in enumerate(self.layers_up):
            if idx == 0:
                x, H, W = layer(x, H, W)
                feature = x.view(B, H, W, -1)
                features.append(feature)
            else:
                x = torch.cat([x, skip_connection[3-idx]], -1)
                x = self.concat_back[idx](x)
                x, H, W = layer(x, H, W)
                feature = x.view(B, H, W, -1)
                if idx < self.num_layers-1:
                    features.append(feature)
        x = self.norm_up(x)
        x, H, W = self.feature_X4(x, H, W)
        feature = x.view(B, H, W, -1)
        features.append(feature)
        # 注意改变维度的摆放位置，L和HW的分离

        feature_map=[]
        feature_map.append(features[3].permute(0, 3, 1, 2))
        feature_map.append(features[2].permute(0, 3, 1, 2))
        feature_map.append(features[1].permute(0, 3, 1, 2))
        feature_map.append(features[0].permute(0, 3, 1, 2))

        uct_map = []
        uct_map.append(self.utc3(feature_map[0]))
        uct_map.append(self.utc2(feature_map[1]))
        uct_map.append(self.utc1(feature_map[2]))
        uct_map.append(self.utc0(feature_map[3]))

        return feature_map, uct_map


if __name__ == '__main__':

    # 模型加载
    model = SwinUnet()
    
    # 预训练权重的载入
    pretrained_path = "/home/lys/Workplace/python/Swin-Unet/pretrained_ckpt/swin_tiny_patch4_window7_224.pth"
    if pretrained_path is not None:
        print("Pretrained_path:{}".format(pretrained_path))
        pretrained_dict = torch.load(pretrained_path, map_location="cpu")
        pretrained_dict = pretrained_dict['model']
        print("Start loading pretrained modle of swin encoder")

        model_dict = model.state_dict()    # 为要赋予的权重
        full_dict = copy.deepcopy(pretrained_dict)  # 为经过预训练的权重
        for k, v in pretrained_dict.items():
            if "layers." in k:
                current_layer_num = 3-int(k[7:8])
                current_k = "layers_up." + str(current_layer_num) + k[8:]
                full_dict.update({current_k:v})
        for k in list(full_dict.keys()):
            if k in model_dict:
                if full_dict[k].shape != model_dict[k].shape:
                    print("delete:{};shape pretrain:{};shape model:{}".format(k,v.shape,model_dict[k].shape))
                    del full_dict[k]

        model.load_state_dict(full_dict, strict=False)
        print("Pre-training weights imported successfully")

    # 测试模型的可用性
    x = torch.randn((2, 3, 512, 512))
    feature,uct_map = model(x)
    for i in feature:
        print(i.shape)
    for i in uct_map:
        print(i.shape)

    # print(y.shape)
