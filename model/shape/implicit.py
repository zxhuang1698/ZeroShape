import numpy as np
import torch
import torch.nn as nn

from functools import partial
from utils.layers import get_embedder
from utils.layers import LayerScale
from timm.models.vision_transformer import Mlp, DropPath
from utils.pos_embed import get_2d_sincos_pos_embed
    
class ImplFuncAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0., last_layer=False):
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.last_layer = last_layer

    def forward(self, x, N_points):
        
        B, N, C = x.shape
        N_latent = N - N_points
        # [3, B, num_heads, N, C/num_heads]
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        # [B, num_heads, N, C/num_heads]
        q, k, v = qkv.unbind(0)
        # [B, num_heads, N_latent, C/num_heads]
        q_latent, k_latent, v_latent = q[:, :, :-N_points], k[:, :, :-N_points], v[:, :, :-N_points]
        # [B, num_heads, N_points, C/num_heads]
        q_points, k_points, v_points = q[:, :, -N_points:], k[:, :, -N_points:], v[:, :, -N_points:]
        
        # attention weight for each point, it's only connected to the latent and itself
        # [B, num_heads, N_points, N_latent+1]
        # get the cross attention, [B, num_heads, N_points, N_latent]
        attn_cross = (q_points @ k_latent.transpose(-2, -1)) * self.scale
        # get the attention to self feature, [B, num_heads, N_points, 1]
        attn_self = torch.sum(q_points * k_points, dim=-1, keepdim=True) * self.scale
        # get the normalized attention, [B, num_heads, N_points, N_latent+1]
        attn_joint = torch.cat([attn_cross, attn_self], dim=-1)
        attn_joint = attn_joint.softmax(dim=-1)
        attn_joint = self.attn_drop(attn_joint)
        
        # break it down to weigh and sum the values
        # [B, num_heads, N_points, N_latent] @ [B, num_heads, N_latent, C/num_heads]
        # -> [B, num_heads, N_points, C/num_heads] -> [B, N_points, C]
        sum_cross = (attn_joint[:, :, :, :N_latent] @ v_latent).transpose(1, 2).reshape(B, N_points, C)
        # [B, num_heads, N_points, 1] * [B, num_heads, N_points, C/num_heads]
        # -> [B, num_heads, N_points, C/num_heads] -> [B, N_points, C]
        sum_self = (attn_joint[:, :, :, N_latent:] * v_points).transpose(1, 2).reshape(B, N_points, C)
        # [B, N_points, C]
        output_points = sum_cross + sum_self
        
        if self.last_layer:
            output = self.proj(output_points)
            output = self.proj_drop(output)
            # [B, N_points, C], [B, N_points, N_latent]
            return output, attn_joint[..., :-1].mean(dim=1)
        
        # attention weight for the latent vec, it's not connected to the points
        # [B, num_heads, N_latent, N_latent]
        attn_latent = (q_latent @ k_latent.transpose(-2, -1)) * self.scale
        attn_latent = attn_latent.softmax(dim=-1)
        attn_latent = self.attn_drop(attn_latent)
        # get the output latent, [B, N_latent, C]
        output_latent = (attn_latent @ v_latent).transpose(1, 2).reshape(B, N_latent, C)
        
        # concatenate the output and return
        output = torch.cat([output_latent, output_points], dim=1)
        output = self.proj(output)
        output = self.proj_drop(output)
        
        # [B, N, C], [B, N_points, N_latent+1]
        return output, attn_joint[..., :-1].mean(dim=1)

class ImplFuncBlock(nn.Module):

    def __init__(
            self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0., init_values=None,
            drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, last_layer=False):
        super().__init__()
        self.last_layer = last_layer
        self.norm1 = norm_layer(dim)
        self.attn = ImplFuncAttention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop, last_layer=last_layer)
        self.ls1 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path1 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.ls2 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path2 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x, unseen_size):
        if self.last_layer:
            attn_out, attn_vis = self.attn(self.norm1(x), unseen_size)
            output = x[:, -unseen_size:] + self.drop_path1(self.ls1(attn_out))
            output = output + self.drop_path2(self.ls2(self.mlp(self.norm2(output))))
            return output, attn_vis
        else:
            attn_out, attn_vis = self.attn(self.norm1(x), unseen_size)
            x = x + self.drop_path1(self.ls1(attn_out))
            x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
            return x, attn_vis

class LinearProj3D(nn.Module):
    """ 
    Linear projection of 3D point into embedding space
    """
    def __init__(self, embed_dim, posenc_res=0):
        super().__init__()
        self.embed_dim = embed_dim
        
        # define positional embedder
        self.embed_fn = None
        input_ch = 3
        if posenc_res > 0:
            self.embed_fn, input_ch = get_embedder(posenc_res, input_dims=3)
        
        # linear proj layer
        self.proj = nn.Linear(input_ch, embed_dim)

    def forward(self, points_3D):
        if self.embed_fn is not None:
            points_3D = self.embed_fn(points_3D)
        return self.proj(points_3D)

class MLPBlocks(nn.Module):
    def __init__(self, num_hidden_layers, n_channels, latent_dim, 
                 skip_in=[], posenc_res=0):
        super().__init__()
        
        # projection to the same number of channels
        self.dims = [3 + latent_dim] + [n_channels] * num_hidden_layers + [1]
        self.num_layers = len(self.dims)
        self.skip_in = skip_in

        # define positional embedder
        self.embed_fn = None
        if posenc_res > 0:
            embed_fn, input_ch = get_embedder(posenc_res, input_dims=3)
            self.embed_fn = embed_fn
            self.dims[0] += (input_ch - 3)

        self.layers = nn.ModuleList([])
        
        for l in range(0, self.num_layers - 1):
            out_dim = self.dims[l + 1]
            if l in self.skip_in:
                in_dim = self.dims[l] + self.dims[0]
            else:
                in_dim = self.dims[l]
                
            lin = nn.Linear(in_dim, out_dim)
            self.layers.append(lin)
        
        # register for param init
        self.posenc_res = posenc_res

        # activation
        self.softplus = nn.Softplus(beta=100)

    def forward(self, points, proj_latent):
        
        # positional encoding
        if self.embed_fn is not None:
            points = self.embed_fn(points)

        # forward by layer
        # [B, N, posenc+C]
        inputs = torch.cat([points, proj_latent], dim=-1)
        x = inputs
        for l in range(0, self.num_layers - 1):
            if l in self.skip_in:
                x = torch.cat([x, inputs], -1) / np.sqrt(2)
            x = self.layers[l](x)
            if l < self.num_layers - 2:
                x = self.softplus(x)
        return x

class Implicit(nn.Module):
    """ 
    Implicit function conditioned on depth encodings
    """
    def __init__(self,
                 num_patches, latent_dim=768, semantic=False, n_channels=512,
                 n_blocks_attn=2, n_layers_mlp=6, num_heads=16, posenc_3D=0,
                 mlp_ratio=4., norm_layer=partial(nn.LayerNorm, eps=1e-6), drop_path=0.1,
                 skip_in=[], pos_perlayer=True):
        super().__init__()
        self.num_patches = num_patches
        self.pos_perlayer = pos_perlayer
        self.semantic = semantic
        
        # projection to the same number of channels, no posenc
        self.point_proj = LinearProj3D(n_channels)
        self.latent_proj = nn.Linear(latent_dim, n_channels, bias=True)
        
        # positional embedding for the depth latent codes
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, n_channels), requires_grad=False)  # fixed sin-cos embedding

        # multi-head attention blocks
        self.blocks_attn = nn.ModuleList([
            ImplFuncBlock(
                n_channels, num_heads, mlp_ratio, 
                qkv_bias=True, norm_layer=norm_layer, drop_path=drop_path
            ) for _ in range(n_blocks_attn-1)])
        self.blocks_attn.append(
            ImplFuncBlock(
                n_channels, num_heads, mlp_ratio, 
                qkv_bias=True, norm_layer=norm_layer, drop_path=drop_path, last_layer=True
            )
        )
        self.norm = norm_layer(n_channels)
        
        self.impl_mlp = None
        # define the impl MLP
        if n_layers_mlp > 0:
            self.impl_mlp = MLPBlocks(n_layers_mlp, n_channels, n_channels, 
                skip_in=skip_in, posenc_res=posenc_3D)
        else:
            # occ and color prediction
            self.pred_head = nn.Linear(n_channels, 1, bias=True)

        self.initialize_weights()

    def initialize_weights(self):
        
        # initialize the positional embedding for the depth latent codes
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.num_patches**.5), cls_token=True)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, latent_depth, latent_semantic, points_3D):
        # concatenate latent codes if semantic is used
        latent = torch.cat([latent_depth, latent_semantic], dim=-1) if self.semantic else latent_depth

        # project latent code and add posenc
        # [B, 1+n_patches, C]
        latent = self.latent_proj(latent)
        N_latent = latent.shape[1]

        # project query points
        # [B, n_points, C_dec]
        points_feat = self.point_proj(points_3D)
        
        # concat point feat with latent
        # [B, 1+n_patches+n_points, C_dec]
        output = torch.cat([latent, points_feat], dim=1)

        # apply multi-head attention blocks
        attn_vis = []
        for l, blk in enumerate(self.blocks_attn):
            if self.pos_perlayer or l == 0:
                output[:, :N_latent] = output[:, :N_latent] + self.pos_embed
            output, attn = blk(output, points_feat.shape[1])
            attn_vis.append(attn)
        output = self.norm(output)
        # average of attention weights across layers, [B, N_points, N_latent+1]
        attn_vis = torch.stack(attn_vis, dim=-1).mean(dim=-1)
        
        if self.impl_mlp:
            # apply mlp blocks
            output = self.impl_mlp(points_3D, output)
        else:
            # predictor projection
            # [B, n_points, 1]
            output = self.pred_head(output)

        # return the occ logit of shape [B, n_points] and the attention weights if needed
        return output.squeeze(-1), attn_vis
