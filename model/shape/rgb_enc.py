# This source code is written based on https://github.com/facebookresearch/MCC 
# The original code base is licensed under the license found in the LICENSE file in the root directory.

import torch
import torch.nn as nn
import torchvision

from functools import partial
from timm.models.vision_transformer import Block, PatchEmbed
from utils.pos_embed import get_2d_sincos_pos_embed
from utils.layers import Bottleneck_Conv

class RGBEncAtt(nn.Module):
    """ 
    Seen surface encoder based on transformer.
    """
    def __init__(self,
                 img_size=224, embed_dim=768, n_blocks=12, num_heads=12, win_size=16,
                 mlp_ratio=4., norm_layer=partial(nn.LayerNorm, eps=1e-6), drop_path=0.1):
        super().__init__()

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.rgb_embed = PatchEmbed(img_size, win_size, 3, embed_dim)
        
        num_patches = self.rgb_embed.num_patches
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim), requires_grad=False)

        self.blocks = nn.ModuleList([
            Block(
                embed_dim, num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer,
                drop_path=drop_path
            ) for _ in range(n_blocks)])

        self.norm = norm_layer(embed_dim)

        self.initialize_weights()

    def initialize_weights(self):
        # initialize the pos enc with fixed cos-sin pattern
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.rgb_embed.num_patches**.5), cls_token=True)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))
        
        # initialize rgb patch_embed like nn.Linear (instead of nn.Conv2d)
        w = self.rgb_embed.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        
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

    def forward(self, rgb_obj):
        
        # [B, H/ws*W/ws, C]
        rgb_embedding = self.rgb_embed(rgb_obj)
        rgb_embedding = rgb_embedding + self.pos_embed[:, 1:, :]

        # append cls token
        # [1, 1, C]
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        # [B, 1, C]
        cls_tokens = cls_token.expand(rgb_embedding.shape[0], -1, -1)

        # [B, H/ws*W/ws+1, C]
        rgb_embedding = torch.cat((cls_tokens, rgb_embedding), dim=1)
        
        # apply Transformer blocks
        for blk in self.blocks:
            rgb_embedding = blk(rgb_embedding)
        rgb_embedding = self.norm(rgb_embedding)

        # [B, H/ws*W/ws+1, C]
        return rgb_embedding

class RGBEncRes(nn.Module):
    """ 
    RGB encoder based on resnet.
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
        self.rgb_feature = None
        def feature_hook(model, input, output):
            self.rgb_feature = output
        
        # attach hooks
        if (opt.arch.win_size) == 16:
            self.encoder.layer3.register_forward_hook(feature_hook)
            self.rgb_feat_proj = nn.Sequential(
                Bottleneck_Conv(1024),
                Bottleneck_Conv(1024),
                nn.Conv2d(1024, opt.arch.latent_dim, 1)
            )
        elif (opt.arch.win_size) == 32:
            self.encoder.layer4.register_forward_hook(feature_hook)
            self.rgb_feat_proj = nn.Sequential(
                Bottleneck_Conv(2048),
                Bottleneck_Conv(2048),
                nn.Conv2d(2048, opt.arch.latent_dim, 1)
            )
        else:
            print('Make sure win_size is 16 or 32 when using resnet backbone!')
            raise NotImplementedError
        
    def forward(self, rgb_obj):
        batch_size = rgb_obj.shape[0]
        assert len(rgb_obj.shape) == 4
        
        # [B, 1, C]
        global_feat = self.encoder(rgb_obj).unsqueeze(1)
        # [B, C, H/ws*W/ws]
        local_feat = self.rgb_feat_proj(self.rgb_feature).view(batch_size, global_feat.shape[-1], -1)
        # [B, H/ws*W/ws, C]
        local_feat = local_feat.permute(0, 2, 1).contiguous()
        # [B, 1+H/ws*W/ws, C]
        rgb_embedding = torch.cat([global_feat, local_feat], dim=1)
        
        return rgb_embedding
