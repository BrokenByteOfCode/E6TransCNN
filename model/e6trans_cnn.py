import torch
import torch.nn as nn
from model.config import E6TransCNNConfig
from model.backbone.stem import E6TransStem
from model.backbone.transformer import E6TransEncoder
from model.ssl.masking import MAEMasker
from model.ssl.decoder import MAEDecoder
from model.heads.classification import RatingHead, GenderHead, GeneralHead, MetaHead
from model.heads.metric import SpeciesMetricHead

class E6TransCNN(nn.Module):
    def __init__(self, config: E6TransCNNConfig):
        super().__init__()
        self.config = config
        self.spatial_size = config.spatial_size
        
        self.stem = E6TransStem(
            in_channels=config.in_channels,
            out_channels=512
        )
        
        self.encoder = E6TransEncoder(
            embed_dim=config.embed_dim,
            depth=config.encoder_depth,
            num_heads=config.encoder_heads,
            mlp_ratio=config.mlp_ratio,
            spatial_size=self.spatial_size,
            in_features=512
        )
        
        self.masker = MAEMasker(
            spatial_size=self.spatial_size,
            mask_ratio=config.mask_ratio
        )
        
        self.decoder = MAEDecoder(
            embed_dim=config.embed_dim,
            decoder_dim=config.decoder_dim,
            decoder_depth=config.decoder_depth,
            decoder_heads=config.decoder_heads,
            spatial_size=self.spatial_size,
            patch_size=16
        )
        
        self.rating_head = RatingHead(
            embed_dim=config.embed_dim,
            num_classes=config.num_classes_rating
        )
        self.gender_head = GenderHead(
            embed_dim=config.embed_dim,
            num_classes=config.num_classes_gender
        )
        self.species_head = SpeciesMetricHead(
            embed_dim=config.embed_dim,
            num_classes=config.num_classes_species
        )
        self.general_head = GeneralHead(
            embed_dim=config.embed_dim,
            num_classes=config.num_classes_general
        )
        self.meta_head = MetaHead(
            embed_dim=config.embed_dim,
            num_classes=config.num_classes_meta
        )

    def forward_ssl(self, imgs):
        x = self.stem(imgs)
        
        B, C, H, W = x.shape
        x_flat = x.flatten(2).transpose(1, 2)
        
        ids_keep, mask, ids_restore = self.masker(x_flat)
        
        encoded = self.encoder(x, ids_keep=ids_keep)
        
        pred = self.decoder(encoded, ids_restore)
        target = self.decoder.patchify(imgs)
        
        return pred, mask, target

    def forward_classifier(self, imgs):
        x = self.stem(imgs)
        
        encoded = self.encoder(x)
        
        pooled = torch.mean(encoded, dim=1)
        
        logits_rating = self.rating_head(pooled)
        logits_gender = self.gender_head(pooled)
        logits_species = self.species_head(pooled)
        logits_general = self.general_head(pooled)
        logits_meta = self.meta_head(pooled)
        
        return {
            "rating": logits_rating,
            "gender": logits_gender,
            "species": logits_species,
            "general": logits_general,
            "meta": logits_meta
        }