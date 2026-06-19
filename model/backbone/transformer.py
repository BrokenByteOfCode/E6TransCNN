import torch
import torch.nn as nn
from model.ssl.masking import get_2d_sincos_pos_embed

class E6TransEncoder(nn.Module):
    def __init__(self, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4.0, spatial_size=51, in_features=512):
        super().__init__()
        self.spatial_size = spatial_size
        self.num_tokens = spatial_size * spatial_size
        
        self.proj = nn.Linear(in_features, embed_dim, bias=True)
        
        self.pos_embed = nn.Parameter(
            torch.zeros(1, self.num_tokens, embed_dim), 
            requires_grad=False
        )
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=int(embed_dim * mlp_ratio),
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True
        )
        self.blocks = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        self.norm = nn.LayerNorm(embed_dim)
        
        self._init_weights()

    def _init_weights(self):
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], self.spatial_size)
        self.pos_embed.data.copy_(pos_embed.unsqueeze(0))
        torch.nn.init.xavier_uniform_(self.proj.weight)

    def forward(self, x, ids_keep=None):
        B, C, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        
        x = self.proj(x)
        x = x + self.pos_embed
        
        if ids_keep is not None:
            x = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, x.shape[2]))
            
        x = self.blocks(x)
        x = self.norm(x)
        return x