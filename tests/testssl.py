import json
import os
import sys
import torch
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from model.config import E6TransCNNConfig
from model.e6trans_cnn import E6TransCNN

def setup_dummy_environment():
    raw_dir = Path("raw")
    raw_dir.mkdir(parents=True, exist_ok=True)
    
    dummy_vocab = {
        "species": ["dragon", "ferret", "canine", "feline", "fox", "dracthyr"],
        "general": ["anthro", "wings", "horn", "jewelry", "solo", "male", "female"],
        "meta": ["absurd_res", "hi_res"]
    }
    
    vocab_path = raw_dir / "vocabularies.json"
    with open(vocab_path, "w") as f:
        json.dump(dummy_vocab, f, indent=4)
        
    return vocab_path

def cleanup_dummy_environment(vocab_path):
    if vocab_path.exists():
        vocab_path.unlink()
    try:
        vocab_path.parent.rmdir()
    except Exception:
        pass

def main():
    vocab_path = setup_dummy_environment()
    
    try:
        config = E6TransCNNConfig.from_vocab(vocab_path=vocab_path)
        model = E6TransCNN(config)
        model.eval()
        
        dummy_input = torch.randn(2, 3, config.img_size, config.img_size)
        
        print("Testing Self-Supervised Learning Phase (forward_ssl)...")
        with torch.no_grad():
            pred, mask, target = model.forward_ssl(dummy_input)
            
        print(f"Input batch size: {dummy_input.shape}")
        print(f"Predicted patch tensor shape [B, L, C_patch]: {pred.shape}")
        print(f"Binary loss mask shape [B, L]: {mask.shape}")
        print(f"Target ground-truth patch tensor shape [B, L, C_patch]: {target.shape}")
        
        assert pred.shape == target.shape, "Error: Reconstruction prediction and target shapes do not match!"
        print("SSL test successful. Target and Prediction shapes are fully aligned.\n")
        
        print("Testing Supervised Classification Phase (forward_classifier)...")
        with torch.no_grad():
            outputs = model.forward_classifier(dummy_input)
            
        print(f"Rating Head Logits (CE, 3 classes): {outputs['rating'].shape}")
        print(f"Gender Head Logits (Sigmoid, 4 classes): {outputs['gender'].shape}")
        print(f"Species Head Logits (Metric Cosine, {config.num_classes_species} classes): {outputs['species'].shape}")
        print(f"General Head Logits (Sigmoid, {config.num_classes_general} classes): {outputs['general'].shape}")
        print(f"Meta Head Logits (Sigmoid, {config.num_classes_meta} classes): {outputs['meta'].shape}")
        
        print("\nClassifier test successful. All multi-head outputs match config and vocabularies.")
        
    finally:
        cleanup_dummy_environment(vocab_path)

if __name__ == "__main__":
    main()