import argparse
import logging
import os
from pathlib import Path
import sys
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from model.config import E6TransCNNConfig
from model.e6trans_cnn import E6TransCNN
from model.utils.dataset import E6TransDataset
from model.losses.asymmetric import AsymmetricLoss
from model.losses.dynamic import UncertaintyLossBalancer

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

def get_args():
    parser = argparse.ArgumentParser(description="E6TransCNN Training Pipeline")
    parser.add_argument("--phase", type=str, required=True, choices=["ssl", "classifier"])
    parser.add_argument("--data_dir", type=str, default="dataset/raw")
    parser.add_argument("--vocab_path", type=str, default="dataset/vocab/vocabularies.json")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints")
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=10)
    return parser.parse_args()

def calculate_mae_loss(pred, target, mask):
    loss = (pred - target) ** 2
    loss = loss.mean(dim=-1)
    loss = (loss * mask).sum() / mask.sum()
    return loss

def train_epoch_ssl(model, dataloader, optimizer, scaler, device):
    model.train()
    total_loss = 0.0
    
    progress_bar = tqdm(dataloader, desc="Train SSL", leave=False)
    for batch_idx, batch in enumerate(progress_bar):
        imgs = batch["image"].to(device)
        optimizer.zero_grad()
        
        with torch.cuda.amp.autocast():
            pred, mask, target = model.forward_ssl(imgs)
            loss = calculate_mae_loss(pred, target, mask)
            
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        total_loss += loss.item()
        progress_bar.set_postfix(
            loss=f"{loss.item():.6f}",
            avg=f"{total_loss / (batch_idx + 1):.6f}"
        )
        
    return total_loss / len(dataloader)

def val_epoch_ssl(model, dataloader, device):
    model.eval()
    total_loss = 0.0
    
    with torch.no_grad():
        progress_bar = tqdm(dataloader, desc="Val SSL", leave=False)
        for batch_idx, batch in enumerate(progress_bar):
            imgs = batch["image"].to(device)
            pred, mask, target = model.forward_ssl(imgs)
            loss = calculate_mae_loss(pred, target, mask)
            total_loss += loss.item()
            progress_bar.set_postfix(
                loss=f"{loss.item():.6f}",
                avg=f"{total_loss / (batch_idx + 1):.6f}"
            )
            
    return total_loss / len(dataloader)

def train_epoch_classifier(model, dataloader, optimizer, scaler, balancer, criterion_asl, criterion_bce, criterion_ce, device):
    model.train()
    balancer.train()
    total_loss = 0.0
    
    progress_bar = tqdm(dataloader, desc="Train Classifier", leave=False)
    for batch in progress_bar:
        imgs = batch["image"].to(device)
        rating_targets = batch["rating"].to(device)
        gender_targets = batch["gender"].to(device)
        species_targets = batch["species"].to(device)
        general_targets = batch["general"].to(device)
        meta_targets = batch["meta"].to(device)
        
        optimizer.zero_grad()
        
        with torch.cuda.amp.autocast():
            outputs = model.forward_classifier(imgs)
            
            loss_rating = criterion_ce(outputs["rating"], rating_targets)
            loss_gender = criterion_bce(outputs["gender"], gender_targets)
            loss_species = criterion_bce(outputs["species"], species_targets)
            loss_general = criterion_asl(outputs["general"], general_targets)
            loss_meta = criterion_asl(outputs["meta"], meta_targets)
            
            losses = [loss_rating, loss_gender, loss_species, loss_general, loss_meta]
            joint_loss = balancer(losses)
            
        scaler.scale(joint_loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        total_loss += joint_loss.item()
        progress_bar.set_postfix(
            joint=f"{joint_loss.item():.4f}",
            rating=f"{loss_rating.item():.4f}",
            gender=f"{loss_gender.item():.4f}",
            species=f"{loss_species.item():.4f}",
            general=f"{loss_general.item():.4f}",
            meta=f"{loss_meta.item():.4f}"
        )
        
    return total_loss / len(dataloader)

def val_epoch_classifier(model, dataloader, balancer, criterion_asl, criterion_bce, criterion_ce, device):
    model.eval()
    balancer.eval()
    total_loss = 0.0
    
    with torch.no_grad():
        progress_bar = tqdm(dataloader, desc="Val Classifier", leave=False)
        for batch in progress_bar:
            imgs = batch["image"].to(device)
            rating_targets = batch["rating"].to(device)
            gender_targets = batch["gender"].to(device)
            species_targets = batch["species"].to(device)
            general_targets = batch["general"].to(device)
            meta_targets = batch["meta"].to(device)
            
            outputs = model.forward_classifier(imgs)
            
            loss_rating = criterion_ce(outputs["rating"], rating_targets)
            loss_gender = criterion_bce(outputs["gender"], gender_targets)
            loss_species = criterion_bce(outputs["species"], species_targets)
            loss_general = criterion_asl(outputs["general"], general_targets)
            loss_meta = criterion_asl(outputs["meta"], meta_targets)
            
            losses = [loss_rating, loss_gender, loss_species, loss_general, loss_meta]
            joint_loss = balancer(losses)
            
            total_loss += joint_loss.item()
            progress_bar.set_postfix(
                joint=f"{joint_loss.item():.4f}",
                rating=f"{loss_rating.item():.4f}",
                gender=f"{loss_gender.item():.4f}",
                species=f"{loss_species.item():.4f}",
                general=f"{loss_general.item():.4f}",
                meta=f"{loss_meta.item():.4f}"
            )
            
    return total_loss / len(dataloader)

def main():
    args = get_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using execution device: {device}")
    
    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    config = E6TransCNNConfig.from_vocab(vocab_path=args.vocab_path)
    model = E6TransCNN(config).to(device)
    
    train_dataset = E6TransDataset(
        data_dir=args.data_dir,
        vocab_path=args.vocab_path,
        split="train",
        img_size=config.img_size
    )
    val_dataset = E6TransDataset(
        data_dir=args.data_dir,
        vocab_path=args.vocab_path,
        split="val",
        img_size=config.img_size
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    balancer = UncertaintyLossBalancer(num_tasks=5).to(device)
    
    if args.phase == "ssl":
        params = list(model.parameters())
    else:
        params = list(model.parameters()) + list(balancer.parameters())
        
    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.05)
    scaler = torch.cuda.amp.GradScaler()
    
    criterion_asl = AsymmetricLoss()
    criterion_bce = nn.BCEWithLogitsLoss()
    criterion_ce = nn.CrossEntropyLoss()
    
    start_epoch = 0
    best_loss = float("inf")
    patience_counter = 0
    
    if args.resume:
        if os.path.exists(args.resume):
            logging.info(f"Resuming training state from checkpoint: {args.resume}")
            checkpoint = torch.load(args.resume, map_location=device)
            
            model.load_state_dict(checkpoint["model_state_dict"])
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            scaler.load_state_dict(checkpoint["scaler_state_dict"])
            
            if "balancer_state_dict" in checkpoint and args.phase == "classifier":
                balancer.load_state_dict(checkpoint["balancer_state_dict"])
                
            start_epoch = checkpoint["epoch"] + 1
            best_loss = checkpoint["best_loss"]
            patience_counter = checkpoint.get("patience_counter", 0)
            logging.info(f"Loaded checkpoint state successfully. Resuming at epoch {start_epoch}")
        else:
            logging.error(f"Checkpoint file path not found: {args.resume}")
            sys.exit(1)
            
    for epoch in range(start_epoch, args.epochs):
        logging.info(f"Starting Epoch {epoch + 1}/{args.epochs}")
        
        if args.phase == "ssl":
            train_loss = train_epoch_ssl(model, train_loader, optimizer, scaler, device)
            val_loss = val_epoch_ssl(model, val_loader, device)
        else:
            train_loss = train_epoch_classifier(
                model, train_loader, optimizer, scaler, balancer,
                criterion_asl, criterion_bce, criterion_ce, device
            )
            val_loss = val_epoch_classifier(
                model, val_loader, balancer,
                criterion_asl, criterion_bce, criterion_ce, device
            )
            
        logging.info(f"Epoch {epoch + 1} completed. Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")
        
        checkpoint_state = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scaler_state_dict": scaler.state_dict(),
            "balancer_state_dict": balancer.state_dict() if args.phase == "classifier" else None,
            "best_loss": best_loss,
            "patience_counter": patience_counter
        }
        
        latest_path = checkpoint_dir / f"checkpoint_{args.phase}_latest.pth"
        torch.save(checkpoint_state, latest_path)
        
        if val_loss < best_loss:
            best_loss = val_loss
            patience_counter = 0
            best_path = checkpoint_dir / f"checkpoint_{args.phase}_best.pth"
            torch.save(checkpoint_state, best_path)
            logging.info(f"Validation loss decreased. Saving best checkpoint at: {best_path}")
        else:
            patience_counter += 1
            logging.info(f"No validation improvement. Early stopping counter: {patience_counter}/{args.patience}")
            
        if patience_counter >= args.patience:
            logging.warning(f"Early stopping triggered. Training stopped at epoch {epoch + 1} to prevent overbaking.")
            break

if __name__ == "__main__":
    main()