import torch
import torch.nn as nn
from functools import partial
from einops import rearrange, repeat
from timm.models.layers import DropPath, to_2tuple, trunc_normal_


class Mlp(nn.Module):

    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        # super()继承父类的构造函数
        super().__init__()
        # 若传入了out_features（格式）,就用传入的，否则就用in_feature
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        # 第一层全连接神经网络，输入层到隐藏层
        self.fc1 = nn.Linear(in_features, hidden_features)
        # 激活函数
        self.act = act_layer()
        # 第二层全连接神经网络，隐藏层到输出层
        self.fc2 = nn.Linear(hidden_features, out_features)
        # ***dropout，训练时常用，可以避免过拟合，增强泛化能力
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        # todo 第一次激活函数后也进行一次，避免过拟合
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


# 多头自注意，定义8个头
class Attention(nn.Module):
    def __init__(self,
                 dim,
                 num_heads=4,
                 qkv_bias=False,
                 qk_scale=None, attn_drop=0.,
                 proj_drop=0.):
        # # super()继承父类的构造函数
        super().__init__()
        self.num_heads = num_heads
        # 对每一个head的dim
        head_dim = dim // num_heads
        # todo NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        # 如果传入了qk_scale,就用传入的，
        # 否则使用每个头维度乘-0.5次方，对应attention公式中（QKT/根号d）*V的分母
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        # 再定义一个全连接层，对应多头自注意输出b Concate拼接后，乘的W0 。输入输出节点个数都等于dim
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)

        #  ####因为前面已经将数据按每个heads进行划分了，所以这里的操作均是对每个heads的qkv进行操作的
        q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple)
        qkv = (q @ k.transpose(-2, -1)) * self.scale
        # dim=-1，就是对矩阵的每一行进行一个softmax处理
        qkv = qkv.softmax(dim=-1)
        # 对得到的结果，也就是V的权重进行dropout
        qkv = self.attn_drop(qkv)

        x = (qkv @ v).transpose(1, 2).reshape(B, N, C)
        # 拼接起来后还需要通过W对其进行映射，所以这里通过proj这个全连接层得到x
        x = self.proj(x)
        # 在进行一次dropout得到最终输出
        x = self.proj_drop(x)
        return x

# Encoder Block
class Block(nn.Module):
    def __init__(self,
                 # 针对每个token的维度
                 dim,
                 num_heads,
                 # 第一个全连接层节点个数是输入节点个数的四倍
                 mlp_ratio=4.,
                 qkv_bias=False,
                 # 默认为None,对应前面q*kt *的self.scale
                 qk_scale=None,
                 # 对应多头自注意模块里的drop_ratio最后全连接层后使用的drop_ratio
                 drop=0.,
                 # 对应q*kt /根号d之后通过softmax之后的dropout层
                 attn_drop=0.,
                 # 对应从多头自注意出来后的,和从MLP Block出来后的DropPath
                 drop_path=0.,
                 # 激活函数默认GELU
                 act_layer=nn.GELU,
                 # 标准化默认采用LayerNorm
                 norm_layer=nn.LayerNorm):
        super().__init__()
        # norm1对应第一个LayerNorm
        self.norm1 = norm_layer(dim)
        # todo 在类中调用同一文件下另一个类
        # 调用前面定义的Attention类，实例化多头自注意模块
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias,
            qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        #   传入的drop_path大于0，实例化DropPath方法；否则，使用nn.Identity()，即不做任何操作。
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        # norm2对应第二个LayerNorm
        self.norm2 = norm_layer(dim)
        # norm2对应第二个LayerNorm
        mlp_hidden_dim = int(dim * mlp_ratio)
        # 初始化MLP结构（维度，隐藏层维数，激活函数，Drop_ratio）
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    # 正向传播过程: x 通过Encoder Block
    def forward(self, x):
        # 第一个残差结构：x经过第一个LayerNorm、多头自注意，Drop,得到的再加上x
        x = x + self.drop_path(self.attn(self.norm1(x)))
        # x经过第二个LayerNorm，MLP,Drop,得到的再加上x
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class SprTransformer(nn.Module):
    # 骨骼点数存疑
    def __init__(self, num_joints=95, in_chans=2, embed_dim_ratio=32, depth=4,
                 num_heads=4, mlp_ratio=2., qkv_bias=True, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.2,  norm_layer=None):

        super().__init__()

        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        embed_dim = embed_dim_ratio * num_joints   # ### temporal embed_dim is num_joints * spatial embedding dim ratio
        # 这里的3是三维吗
        # out_dim = num_joints * 3     # ### output dimension is num_joints * 3

        # ## spatial patch embedding
        self.Spatial_patch_to_embedding = nn.Linear(in_chans, embed_dim_ratio)
        self.Spatial_pos_embed = nn.Parameter(torch.zeros(1, num_joints, embed_dim_ratio))

        # * self.Temporal_pos_embed = nn.Parameter(torch.zeros(1, num_frame, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule

        # 四个Block
        self.Spatial_blocks = nn.ModuleList([
            Block(
                dim=embed_dim_ratio, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer)
            # 随机深度：Encoder的个数
            for i in range(depth)])

        self.Spatial_norm = norm_layer(embed_dim_ratio)

    # 空间前向传播
    """
    x: transformer_features :Tensor(12, 323, 190)
    
    """
    def Spatial_forward_features(self, x):
        # x = x.repeat(2, 1, 1, 1).shape    ((batch, 323, 95, 2))
        b, f, p, c = x.shape  # #### b is batch size, f is number of frames, p is number of joints
        # x = (32*2,9,9)
        x = rearrange(x, 'b f p c -> (b f) p  c')

        # in_chans -> embed_dim_ratio
        x = self.Spatial_patch_to_embedding(x)
        # torch.zeros(1, num_joints, embed_dim_ratio)
        x += self.Spatial_pos_embed
        x = self.pos_drop(x)

        for blk in self.Spatial_blocks:
            x = blk(x)

        x = self.Spatial_norm(x)
        # x = rearrange(x, '(b f) w c -> b f (w c)', f=f)
        return x

    def forward(self, x):
        x = self.Spatial_forward_features(x)

        return x


# model = SprTransformer()
# print(model)
