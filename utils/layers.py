import torch
import torch.nn as nn

from functools import partial
from timm.models.vision_transformer import Block

# 3D positional encoding, from https://github.com/bmild/nerf.
class Embedder:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.create_embedding_fn()

    def create_embedding_fn(self):
        embed_fns = []
        d = self.kwargs['input_dims']
        out_dim = 0
        if self.kwargs['include_input']:
            embed_fns.append(lambda x: x)
            out_dim += d

        max_freq = self.kwargs['max_freq_log2']
        N_freqs = self.kwargs['num_freqs']

        if self.kwargs['log_sampling']:
            freq_bands = 2. ** torch.linspace(0., max_freq, N_freqs)
        else:
            freq_bands = torch.linspace(2.**0., 2.**max_freq, N_freqs)

        for freq in freq_bands:
            for p_fn in self.kwargs['periodic_fns']:
                embed_fns.append(lambda x, p_fn=p_fn,
                                 freq=freq: p_fn(x * freq))
                out_dim += d

        self.embed_fns = embed_fns
        self.out_dim = out_dim

    def embed(self, inputs):
        return torch.cat([fn(inputs) for fn in self.embed_fns], -1)

def get_embedder(posenc_res, input_dims=3):
    embed_kwargs = {
        'include_input': True,
        'input_dims': input_dims,
        'max_freq_log2': posenc_res-1,
        'num_freqs': posenc_res,
        'log_sampling': True,
        'periodic_fns': [torch.sin, torch.cos],
    }

    embedder_obj = Embedder(**embed_kwargs)
    def embed(x, eo=embedder_obj): return eo.embed(x)
    return embed, embedder_obj.out_dim

class LayerScale(nn.Module):
    def __init__(self, dim, init_values=1e-5, inplace=False):
        super().__init__()
        self.inplace = inplace
        self.gamma = nn.Parameter(init_values * torch.ones(dim))

    def forward(self, x):
        return x.mul_(self.gamma) if self.inplace else x * self.gamma

class Bottleneck_Linear(nn.Module):
    def __init__(self, n_channels):
        super().__init__()
        self.linear1 = nn.Linear(n_channels, n_channels)
        self.norm = nn.LayerNorm(n_channels)
        self.linear2 = nn.Linear(n_channels, n_channels)
        self.gelu = nn.GELU()
        
    def forward(self, x):
        x = x + self.linear2(self.gelu(self.linear1(self.norm(x))))
        return x

class Bottleneck_Conv(nn.Module):
    def __init__(self, n_channels, kernel_size=1):
        super().__init__()
        self.linear1 = nn.Conv2d(n_channels, n_channels, kernel_size=kernel_size, padding=kernel_size//2, bias=False)
        self.bn1 = nn.BatchNorm2d(n_channels)
        self.linear2 = nn.Conv2d(n_channels, n_channels, kernel_size=kernel_size, padding=kernel_size//2, bias=False)
        self.bn2 = nn.BatchNorm2d(n_channels)
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x):
        assert len(x.shape) in [2, 4]
        input_dims = len(x.shape)
        if input_dims == 2:
            x = x.unsqueeze(-1).unsqueeze(-1)
        residual = x
        out = self.linear1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.linear2(out)
        out = self.bn2(out)
        out += residual
        out = self.relu(out)
        if input_dims == 2:
            out = out.squeeze(-1).squeeze(-1)
        return out

class CLIPFusionBlock_Concat(nn.Module):
    """ 
    Fuse clip and rgb embeddings via concat-proj
    """
    def __init__(self, n_channels=512, n_layers=1, act=True):
        super().__init__()
        proj = [Bottleneck_Linear(2 * n_channels) for _ in range(n_layers)]
        proj.append(nn.Linear(2 * n_channels, n_channels))
        if act: proj.append(nn.GELU())
        self.proj = nn.Sequential(*proj)
    
    def forward(self, sem_latent, clip_latent):
        """ 
        sem_latent: [B, N, C]
        clip_latent: [B, C]
        """
        # [B, N, 2C]
        latent_concat = torch.cat([sem_latent, clip_latent.unsqueeze(1).expand_as(sem_latent)], dim=-1)
        # [B, N, C]
        latent = self.proj(latent_concat)
        return latent

class CLIPFusionBlock_Attn(nn.Module):
    """ 
    Fuse geometric and semantic embeddings via multi-layer MHA blocks
    """
    def __init__(self, n_channels=512, n_layers=1, act=True):
        super().__init__()
        self.attn_blocks = nn.ModuleList(
            [Block(
                n_channels, 8, 4, qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), drop_path=0.1
            ) for _ in range(n_layers)]
        )
        if act: self.attn_blocks.append(nn.GELU())
    
    def forward(self, sem_latent, clip_latent):
        """ 
        sem_latent: [B, N, C]
        clip_latent: [B, C]
        """
        # [B, 1+N, C], clip first
        latent = torch.cat([clip_latent.unsqueeze(1), sem_latent], dim=1)
        for attn_block in self.attn_blocks:
            latent = attn_block(latent)
        # [B, N, C]
        return latent[:, 1:, :]