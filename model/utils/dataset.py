import os
import json
import random
import logging
import torch
from PIL import Image
from pathlib import Path
from collections import Counter
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

GENDER_VOCAB = ["male", "female", "androgynous", "ambiguous_gender"]

def build_vocabularies(data_dir, output_dir, min_freq=5):
    data_dir = Path(data_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    species_counter = Counter()
    general_counter = Counter()
    meta_counter = Counter()
    
    json_files = list(data_dir.glob("**/*.json"))
    json_files.sort(key=lambda x: x.as_posix())
    logging.info(f"Scanning {len(json_files)} JSON files to build vocabularies...")
    
    for json_path in json_files:
        try:
            with open(json_path, "r") as f:
                meta = json.load(f)
            tags = meta.get("tags", {})
            
            species_counter.update(tags.get("species", []))
            meta_counter.update(tags.get("meta", []))
            
            for tag in tags.get("general", []):
                if tag not in GENDER_VOCAB:
                    general_counter.update([tag])
                    
        except Exception as e:
            logging.error(f"Error parsing {json_path} during vocab build: {e}")
            
    species_vocab = [tag for tag, count in species_counter.items() if count >= min_freq]
    general_vocab = [tag for tag, count in general_counter.items() if count >= min_freq]
    meta_vocab = [tag for tag, count in meta_counter.items() if count >= min_freq]
    
    species_vocab.sort()
    general_vocab.sort()
    meta_vocab.sort()
    
    vocabs = {
        "species": species_vocab,
        "general": general_vocab,
        "meta": meta_vocab
    }
    
    vocab_path = output_dir / "vocabularies.json"
    with open(vocab_path, "w") as f:
        json.dump(vocabs, f, indent=4)
        
    logging.info(f"Vocabularies saved. Species: {len(species_vocab)}, General: {len(general_vocab)}, Meta: {len(meta_vocab)}")
    return vocabs

class E6TransDataset(Dataset):
    def __init__(self, data_dir, vocab_path, split="train", val_split=0.1, img_size=816, seed=42):
        self.data_dir = Path(data_dir)
        self.img_size = img_size
        self.split = split
        
        with open(vocab_path, "r") as f:
            vocabs = json.load(f)
            
        self.species_vocab = {tag: idx for idx, tag in enumerate(vocabs["species"])}
        self.general_vocab = {tag: idx for idx, tag in enumerate(vocabs["general"])}
        self.meta_vocab = {tag: idx for idx, tag in enumerate(vocabs["meta"])}
        self.gender_vocab = {tag: idx for idx, tag in enumerate(GENDER_VOCAB)}
        
        self.rating_map = {"s": 0, "q": 1, "e": 2}
        
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]
        
        all_samples = self._find_valid_samples()
        all_samples.sort(key=lambda x: x[0].as_posix())
        
        random.seed(seed)
        random.shuffle(all_samples)
        
        val_size = int(len(all_samples) * val_split)
        if split == "train":
            self.samples = all_samples[val_size:]
        elif split == "val":
            self.samples = all_samples[:val_size]
        else:
            self.samples = all_samples
            
        logging.info(f"Dataset partition '{split}' loaded with {len(self.samples)} samples.")

    def _find_valid_samples(self):
        valid_samples = []
        json_files = list(self.data_dir.glob("**/*.json"))
        supported_extensions = [".png", ".jpg", ".jpeg", ".webp", ".PNG", ".JPG", ".JPEG", ".WEBP"]
        
        for json_path in json_files:
            try:
                with open(json_path, "r") as f:
                    json.load(f)
            except Exception:
                logging.warning(f"Skipping corrupted metadata file: {json_path}")
                continue
                
            img_found = False
            for ext in supported_extensions:
                img_path = json_path.with_suffix(ext)
                if img_path.exists():
                    try:
                        with Image.open(img_path) as img:
                            img.verify()
                        valid_samples.append((img_path, json_path))
                        img_found = True
                        break
                    except Exception:
                        logging.warning(f"Skipping corrupted image file: {img_path}")
                        
            if not img_found:
                logging.warning(f"No valid image found for metadata: {json_path}")
                
        return valid_samples

    def __len__(self):
        return len(self.samples)

    def _process_image(self, img_path):
        img = Image.open(img_path).convert("RGB")
        w, h = img.size
        scale = self.img_size / max(w, h)
        new_w, new_h = int(w * scale), int(h * scale)
        
        img = img.resize((new_w, new_h), resample=Image.Resampling.LANCZOS)
        
        pad_w = self.img_size - new_w
        pad_h = self.img_size - new_h
        
        pad_left = pad_w // 2
        pad_top = pad_h // 2
        pad_right = pad_w - pad_left
        pad_bottom = pad_h - pad_top
        
        img_tensor = TF.to_tensor(img)
        img_tensor = TF.normalize(img_tensor, mean=self.mean, std=self.std)
        
        img_tensor = TF.pad(
            img_tensor, 
            [pad_left, pad_top, pad_right, pad_bottom], 
            fill=0.0, 
            padding_mode="constant"
        )
        return img_tensor

    def __getitem__(self, idx):
        img_path, json_path = self.samples[idx]
        
        img_tensor = self._process_image(img_path)
        
        with open(json_path, "r") as f:
            meta = json.load(f)
            
        rating = self.rating_map.get(meta.get("rating", "s"), 0)
        tags = meta.get("tags", {})
        general_tags = tags.get("general", [])
        
        species_target = torch.zeros(len(self.species_vocab))
        unmapped_species = 0
        for tag in tags.get("species", []):
            if tag in self.species_vocab:
                species_target[self.species_vocab[tag]] = 1.0
            else:
                unmapped_species += 1
                
        general_target = torch.zeros(len(self.general_vocab))
        unmapped_general = 0
        for tag in general_tags:
            if tag in self.general_vocab:
                general_target[self.general_vocab[tag]] = 1.0
            elif tag not in GENDER_VOCAB:
                unmapped_general += 1
                
        meta_target = torch.zeros(len(self.meta_vocab))
        unmapped_meta = 0
        for tag in tags.get("meta", []):
            if tag in self.meta_vocab:
                meta_target[self.meta_vocab[tag]] = 1.0
            else:
                unmapped_meta += 1
                
        gender_target = torch.zeros(len(self.gender_vocab))
        for tag in general_tags:
            if tag in self.gender_vocab:
                gender_target[self.gender_vocab[tag]] = 1.0
                
        if (unmapped_species + unmapped_general + unmapped_meta) > 10:
            logging.warning(
                f"High count of unmapped tags in {json_path.name}: "
                f"Species: {unmapped_species}, General: {unmapped_general}, Meta: {unmapped_meta}. "
                f"Consider rebuilding vocabulary."
            )
            
        return {
            "image": img_tensor,
            "rating": torch.tensor(rating, dtype=torch.long),
            "species": species_target,
            "general": general_target,
            "meta": meta_target,
            "gender": gender_target
        }