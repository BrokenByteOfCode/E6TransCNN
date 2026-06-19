import json
import logging
from pathlib import Path
from dataclasses import dataclass
from model.utils.dataset import GENDER_VOCAB

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

@dataclass
class E6TransCNNConfig:
    img_size: int = 816
    in_channels: int = 3
    
    embed_dim: int = 768
    encoder_depth: int = 12
    encoder_heads: int = 12
    mlp_ratio: float = 4.0
    
    mask_ratio: float = 0.75
    decoder_depth: int = 4
    decoder_dim: int = 512
    decoder_heads: int = 16
    
    num_classes_rating: int = 3
    num_classes_gender: int = len(GENDER_VOCAB)
    num_classes_species: int = 0
    num_classes_general: int = 0
    num_classes_meta: int = 0

    @property
    def spatial_size(self) -> int:
        return self.img_size // 16

    @classmethod
    def from_vocab(cls, vocab_path, **kwargs):
        vocab_path = Path(vocab_path)
        if not vocab_path.exists():
            raise FileNotFoundError(f"Vocabulary file not found at {vocab_path}. Please run vocabulary build first.")
            
        with open(vocab_path, "r") as f:
            vocabs = json.load(f)
            
        for key in ("species", "general", "meta"):
            if not vocabs.get(key):
                raise ValueError(
                    f"Vocabulary for '{key}' is empty or missing in {vocab_path}. "
                    f"Please check your dataset and adjust min_freq in build_vocabularies."
                )
                
        config_args = {
            "num_classes_species": len(vocabs["species"]),
            "num_classes_general": len(vocabs["general"]),
            "num_classes_meta": len(vocabs["meta"])
        }
        
        protected_keys = {"num_classes_species", "num_classes_general", "num_classes_meta", "num_classes_gender"}
        overrides = {k: v for k, v in kwargs.items() if k not in protected_keys}
        
        config_args.update(overrides)
        return cls(**config_args)