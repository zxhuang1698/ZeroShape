# This source code is written based on https://github.com/facebookresearch/MCC 
# The original code base is licensed under the license found in the LICENSE file in the root directory.

import torch
import torch.nn as nn
import torchvision

from functools import partial
from timm.models.vision_transformer import Block
from utils.pos_embed import get_2d_sincos_pos_embed
from utils.layers import Bottleneck_Conv

class CoordEmb(nn.Module):
    """ 
    Encode the seen coordinate map to a lower resolution feature map
    Achieved with window-wise attention block by deviding coord map into windows
    Each window is seperately encoded into a single CLS token with self-attention and posenc
    """
    def __init__(self, embed_dim, win_size=8, num_heads=8):
        super().__init__()
        self.embed_dim = embed_dim
        self.win_size = win_size

        self.two_d_pos_embed = nn.Parameter(
            torch.zeros(1, self.win_size*self.win_size + 1, embed_dim), requires_grad=False)  # fixed sin-cos embedding

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        self.pos_embed = nn.Linear(3, embed_dim)

        self.blocks = nn.ModuleList([
            # each block is a residual block with layernorm -> attention -> layernorm -> mlp
            Block(embed_dim, num_heads=num_heads, mlp_ratio=2.0, qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6))
            for _ in range(1)
        ])

        self.invalid_coord_token = nn.Parameter(torch.zeros(embed_dim,))

        self.initialize_weights()

    def initialize_weights(self):
        torch.nn.init.normal_(self.cls_token, std=.02)

        two_d_pos_embed = get_2d_sincos_pos_embed(self.two_d_pos_embed.shape[-1], self.win_size, cls_token=True)
        self.two_d_pos_embed.data.copy_(torch.from_numpy(two_d_pos_embed).float().unsqueeze(0))

        torch.nn.init.normal_(self.invalid_coord_token, std=.02)

    def forward(self, coord_obj, mask_obj):
        # [B, H, W, C]
        emb = self.pos_embed(coord_obj)

        emb[~mask_obj] = 0.0
        emb[~mask_obj] += self.invalid_coord_token

        B, H, W, C = emb.shape
        # [B, H/ws, 8, W/ws, W, C]
        emb = emb.view(B, H // self.win_size, self.win_size, W // self.win_size, self.win_size, C)
        # [B * H/ws * W/ws, 64, C]
        emb = emb.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, self.win_size * self.win_size, C)

        # [B * H/ws * W/ws, 64, C], add posenc that is local to each patch
        emb = emb + self.two_d_pos_embed[:, 1:, :]
        # [1, 1, C]
        cls_token = self.cls_token + self.two_d_pos_embed[:, :1, :]

        # [B * H/ws * W/ws, 1, C]
        cls_tokens = cls_token.expand(emb.shape[0], -1, -1)
        # [B * H/ws * W/ws, 65, C]
        emb = torch.cat((cls_tokens, emb), dim=1)
        
        # transformer (single block) that handle each of the patch seperately
        # reasoning is done within each batch
        for _, blk in enumerate(self.blocks):
            emb = blk(emb)
        
        # return the cls token of each window, [B, H/ws*W/ws, C]
        return emb[:, 0].view(B, (H // self.win_size) * (W // self.win_size), -1)

class CoordEncAtt(nn.Module):
    """ 
    Seen surface encoder based on transformer.
    """
    def __init__(self,
                 embed_dim=768, n_blocks=12, num_heads=12, win_size=8,
                 mlp_ratio=4., norm_layer=partial(nn.LayerNorm, eps=1e-6), drop_path=0.1):
        super().__init__()

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.coord_embed = CoordEmb(embed_dim, win_size, num_heads)

        self.blocks = nn.ModuleList([
            Block(
                embed_dim, num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer,
                drop_path=drop_path
            ) for _ in range(n_blocks)])

        self.norm = norm_layer(embed_dim)

        self.initialize_weights()

    def initialize_weights(self):
        # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.)
        torch.nn.init.normal_(self.cls_token, std=.02)

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

    def forward(self, coord_obj, mask_obj):
        
        # [B, H/ws*W/ws, C]
        coord_embedding = self.coord_embed(coord_obj, mask_obj)

        # append cls token
        # [1, 1, C]
        cls_token = self.cls_token
        # [B, 1, C]
        cls_tokens = cls_token.expand(coord_embedding.shape[0], -1, -1)

        # [B, H/ws*W/ws+1, C]
        coord_embedding = torch.cat((cls_tokens, coord_embedding), dim=1)
        
        # apply Transformer blocks
        for blk in self.blocks:
            coord_embedding = blk(coord_embedding)
        coord_embedding = self.norm(coord_embedding)

        # [B, H/ws*W/ws+1, C]
        return coord_embedding

class CoordEncRes(nn.Module):
    """ 
    Seen surface encoder based on resnet.
    """
    def __init__(self, opt):
        super().__init__()

        self.encoder = torchvision.models.resnet50(pretrained=True)
        self.encoder.fc = nn.Sequential(
            Bottleneck_Conv(2048),
            Bottleneck_Conv(2048),
            nn.Linear(2048, opt.arch.latent_dim)
        )
        
        # define hooks
        self.seen_feature = None
        def feature_hook(model, input, output):
            self.seen_feature = output
        
        # attach hooks
        assert opt.arch.depth.dsp == 1
        if (opt.arch.win_size) == 16:
            self.encoder.layer3.register_forward_hook(feature_hook)
            self.depth_feat_proj = nn.Sequential(
                Bottleneck_Conv(1024),
                Bottleneck_Conv(1024),
                nn.Conv2d(1024, opt.arch.latent_dim, 1)
            )
        elif (opt.arch.win_size) == 32:
            self.encoder.layer4.register_forward_hook(feature_hook)
            self.depth_feat_proj = nn.Sequential(
                Bottleneck_Conv(2048),
                Bottleneck_Conv(2048),
                nn.Conv2d(2048, opt.arch.latent_dim, 1)
            )
        else:
            print('Make sure win_size is 16 or 32 when using resnet backbone!')
            raise NotImplementedError
        
    def forward(self, coord_obj, mask_obj):
        batch_size = coord_obj.shape[0]
        assert len(coord_obj.shape) == len(mask_obj.shape) == 4
        mask_obj = mask_obj.float()
        coord_obj = coord_obj * mask_obj
        
        # [B, 1, C]
        global_feat = self.encoder(coord_obj).unsqueeze(1)
        # [B, C, H/ws*W/ws]
        local_feat = self.depth_feat_proj(self.seen_feature).view(batch_size, global_feat.shape[-1], -1)
        # [B, H/ws*W/ws, C]
        local_feat = local_feat.permute(0, 2, 1).contiguous()
        # [B, 1+H/ws*W/ws, C]
        seen_embedding = torch.cat([global_feat, local_feat], dim=1)
        
        return seen_embedding