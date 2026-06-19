import torch
import torch.nn as nn
from model.ssl.masking import get_2d_sincos_pos_embed

class MAEDecoder(nn.Module):
    def __init__(self, embed_dim=768, decoder_dim=512, decoder_depth=4, decoder_heads=16, spatial_size=51, patch_size=16):
        super().__init__()
        self.spatial_size = spatial_size
        self.num_tokens = spatial_size * spatial_size
        self.patch_size = patch_size
        
        self.decoder_embed = nn.Linear(embed_dim, decoder_dim, bias=True)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_dim))
        
        self.decoder_pos_embed = nn.Parameter(
            torch.zeros(1, self.num_tokens, decoder_dim), 
            requires_grad=False
        )
        
        decoder_layer = nn.TransformerEncoderLayer(
            d_model=decoder_dim,
            nhead=decoder_heads,
            dim_feedforward=decoder_dim * 4,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True
        )
        self.decoder_blocks = nn.TransformerEncoder(decoder_layer, num_layers=decoder_depth)
        self.decoder_norm = nn.LayerNorm(decoder_dim)
        
        output_dim = patch_size * patch_size * 3
        self.decoder_pred = nn.Linear(decoder_dim, output_dim, bias=True)
        
        self._init_weights()

    def _init_weights(self):
        pos_embed = get_2d_sincos_pos_embed(self.decoder_pos_embed.shape[-1], self.spatial_size)
        self.decoder_pos_embed.data.copy_(pos_embed.unsqueeze(0))
        
        torch.nn.init.normal_(self.mask_token, std=.02)
        torch.nn.init.xavier_uniform_(self.decoder_embed.weight)
        torch.nn.init.xavier_uniform_(self.decoder_pred.weight)

    def patchify(self, imgs):
        B, C, H, W = imgs.shape
        p = self.patch_size
        h_patches = H // p
        w_patches = W // p
        
        x = imgs.reshape(shape=[B, 3, h_patches, p, w_patches, p])
        x = torch.einsum('nchpwq->nhwpqc', x)
        x = x.reshape(shape=[B, h_patches * w_patches, p * p * 3])
        return x

    def unpatchify(self, x):
        p = self.patch_size
        h_patches = self.spatial_size
        w_patches = self.spatial_size
        
        x = x.reshape(shape=[x.shape[0], h_patches, w_patches, p, p, 3])
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=[x.shape[0], 3, h_patches * p, w_patches * p])
        return imgs

    def forward(self, x, ids_restore):
        x = self.decoder_embed(x)
        
        mask_tokens = self.mask_token.repeat(x.shape[0], ids_restore.shape[1] - x.shape[1], 1)
        x_ = torch.cat([x, mask_tokens], dim=1)
        x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))
        
        x = x_ + self.decoder_pos_embed
        
        x = self.decoder_blocks(x)
        x = self.decoder_norm(x)
        
        x = self.decoder_pred(x)
        return x