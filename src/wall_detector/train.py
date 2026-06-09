"""
Improved training script for Wall Detection UNet model.
Trains on CubiCasa5K dataset with higher resolution and better augmentations.
"""

import os
import sys
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split, Subset
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
import numpy as np

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from src.wall_detector.model import create_model, DiceBCELoss
from src.wall_detector.dataset import (
    CubiCasaWallDataset, SimpleWallDataset,
    get_train_transforms, get_val_transforms
)


def calculate_iou(pred, target, threshold=0.5):
    """Calculate Intersection over Union."""
    pred_binary = (torch.sigmoid(pred) > threshold).float()
    intersection = (pred_binary * target).sum()
    union = pred_binary.sum() + target.sum() - intersection
    if union == 0:
        return 1.0
    return (intersection / union).item()


def train_one_epoch(model, dataloader, criterion, optimizer, device, scaler=None):
    """Train for one epoch with mixed precision."""
    model.train()
    total_loss = 0
    total_iou = 0
    num_batches = 0
    
    for images, masks in dataloader:
        images = images.to(device)
        masks = masks.to(device)
        
        optimizer.zero_grad()
        
        if scaler is not None:
            with torch.amp.autocast('cuda'):
                outputs = model(images)
                loss = criterion(outputs, masks)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(images)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()
        
        total_loss += loss.item()
        total_iou += calculate_iou(outputs, masks)
        num_batches += 1
    
    return total_loss / max(num_batches, 1), total_iou / max(num_batches, 1)


def validate(model, dataloader, criterion, device):
    """Validate the model."""
    model.eval()
    total_loss = 0
    total_iou = 0
    num_batches = 0
    
    with torch.no_grad():
        for images, masks in dataloader:
            images = images.to(device)
            masks = masks.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, masks)
            
            total_loss += loss.item()
            total_iou += calculate_iou(outputs, masks)
            num_batches += 1
    
    return total_loss / max(num_batches, 1), total_iou / max(num_batches, 1)


def train(
    data_dir="data/cubicasa5k",
    output_dir="src/wall_detector/weights",
    image_size=512,
    batch_size=8,
    num_epochs=80,
    learning_rate=3e-4,
    patience=15,
    use_simple_dataset=False,
    encoder_name="resnet34",
):
    """Main training function.
    
    Args:
        data_dir: Path to CubiCasa5K dataset
        output_dir: Where to save model weights
        image_size: Input image size (512 for better detail)
        batch_size: Training batch size
        num_epochs: Max training epochs
        learning_rate: Initial learning rate
        patience: Early stopping patience
        use_simple_dataset: Use threshold-based masks instead of SVG parsing
        encoder_name: Backbone encoder (resnet34 for better accuracy)
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    
    # Create dataset
    print(f"\nLoading dataset from {data_dir}...")
    
    # Auto-detect CubiCasa directory structure
    actual_data_dir = data_dir
    if os.path.exists(os.path.join(data_dir, 'cubicasa5k')):
        actual_data_dir = os.path.join(data_dir, 'cubicasa5k')
    
    if use_simple_dataset:
        # Find all PNG images recursively
        image_paths = []
        for root, dirs, files in os.walk(actual_data_dir):
            for f in files:
                if f.endswith('.png') and ('original' in f.lower() or 'F1' in f):
                    image_paths.append(os.path.join(root, f))
        
        print(f"Found {len(image_paths)} images for simple dataset")
        
        if len(image_paths) == 0:
            print("ERROR: No images found! Check the data directory.")
            return
        
        # Split
        n_train = int(len(image_paths) * 0.85)
        train_paths = image_paths[:n_train]
        val_paths = image_paths[n_train:]
        
        train_dataset = SimpleWallDataset(train_paths, get_train_transforms(image_size), image_size)
        val_dataset = SimpleWallDataset(val_paths, get_val_transforms(image_size), image_size)
    else:
        full_dataset = CubiCasaWallDataset(
            actual_data_dir,
            transform=None,
            image_size=image_size
        )
        
        if len(full_dataset) == 0:
            print("ERROR: No valid samples found! Trying to find in subdirectories...")
            # Try high_quality and colorful subdirectories
            for sub in ['high_quality', 'colorful']:
                sub_dir = os.path.join(actual_data_dir, sub)
                if os.path.exists(sub_dir):
                    full_dataset = CubiCasaWallDataset(
                        sub_dir,
                        transform=None,
                        image_size=image_size
                    )
                    if len(full_dataset) > 0:
                        print(f"Found samples in {sub_dir}")
                        break
            
            if len(full_dataset) == 0:
                print("Falling back to simple dataset...")
                return train(data_dir=data_dir, output_dir=output_dir, use_simple_dataset=True,
                           image_size=image_size, batch_size=batch_size, num_epochs=num_epochs,
                           encoder_name=encoder_name)
        
        # Split: 85% train, 15% val
        n_total = len(full_dataset)
        n_train = int(n_total * 0.85)
        n_val = n_total - n_train
        
        # Use random_split for simplicity
        train_subset, val_subset = random_split(
            full_dataset, [n_train, n_val],
            generator=torch.Generator().manual_seed(42)
        )
        
        # Wrap subsets with proper transforms
        train_dataset = TransformWrapper(train_subset, get_train_transforms(image_size))
        val_dataset = TransformWrapper(val_subset, get_val_transforms(image_size))
    
    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")
    
    # DataLoaders
    num_workers = min(4, os.cpu_count() or 1)
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, drop_last=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )
    
    # Model - use resnet34 for better accuracy
    print(f"\nCreating UNet model with {encoder_name} encoder...")
    model = create_model(encoder_name=encoder_name).to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Loss - use heavier dice weight for better segmentation
    criterion = DiceBCELoss(dice_weight=0.7, bce_weight=0.3)
    
    # Optimizer with weight decay
    optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    
    # Cosine annealing with warm restarts
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2, eta_min=1e-6)
    
    # Mixed precision for faster training
    scaler = torch.amp.GradScaler('cuda') if device.type == 'cuda' else None
    
    # Training loop
    print(f"\nStarting training for {num_epochs} epochs...")
    print(f"Image size: {image_size}x{image_size}")
    print(f"Batch size: {batch_size}")
    print(f"Learning rate: {learning_rate}")
    print("=" * 70)
    
    best_val_loss = float('inf')
    best_val_iou = 0
    epochs_without_improvement = 0
    start_time = time.time()
    
    for epoch in range(num_epochs):
        epoch_start = time.time()
        
        # Train
        train_loss, train_iou = train_one_epoch(model, train_loader, criterion, optimizer, device, scaler)
        
        # Validate
        val_loss, val_iou = validate(model, val_loader, criterion, device)
        
        # Step scheduler
        scheduler.step()
        
        epoch_time = time.time() - epoch_start
        current_lr = optimizer.param_groups[0]['lr']
        
        print(f"Epoch [{epoch+1:3d}/{num_epochs}] "
              f"Train Loss: {train_loss:.4f} IoU: {train_iou:.4f} | "
              f"Val Loss: {val_loss:.4f} IoU: {val_iou:.4f} | "
              f"LR: {current_lr:.2e} | Time: {epoch_time:.1f}s")
        
        # Save best model
        improved = False
        if val_iou > best_val_iou:
            best_val_iou = val_iou
            improved = True
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            improved = True
            
        if improved:
            epochs_without_improvement = 0
            
            save_path = os.path.join(output_dir, 'wall_detector_best.pth')
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'val_iou': val_iou,
                'image_size': image_size,
                'encoder_name': encoder_name,
            }, save_path)
            print(f"  ✓ Saved best model (Val Loss: {val_loss:.4f}, IoU: {val_iou:.4f})")
        else:
            epochs_without_improvement += 1
        
        # Early stopping
        if epochs_without_improvement >= patience:
            print(f"\nEarly stopping at epoch {epoch+1} (no improvement for {patience} epochs)")
            break
    
    total_time = time.time() - start_time
    print("=" * 70)
    print(f"Training complete in {total_time/60:.1f} minutes")
    print(f"Best Val Loss: {best_val_loss:.4f}")
    print(f"Best Val IoU: {best_val_iou:.4f}")
    print(f"Model saved to: {os.path.join(output_dir, 'wall_detector_best.pth')}")
    
    return model


class TransformWrapper:
    """Wraps a Subset with custom transforms applied at __getitem__ time."""
    
    def __init__(self, subset, transform):
        self.subset = subset
        self.transform = transform
    
    def __len__(self):
        return len(self.subset)
    
    def __getitem__(self, idx):
        image, mask = self.subset[idx]
        
        # If already tensors (from CubiCasaWallDataset without transform), convert back
        if isinstance(image, torch.Tensor):
            image = image.numpy().transpose(1, 2, 0)  # CHW -> HWC
            mask = mask.squeeze(0).numpy()
        elif isinstance(image, np.ndarray) and image.dtype != np.uint8:
            image = (image * 255).astype(np.uint8)
            
        if isinstance(mask, torch.Tensor):
            mask = mask.squeeze(0).numpy()
        
        # Apply transforms
        if self.transform:
            transformed = self.transform(image=image, mask=mask)
            image = transformed['image']
            mask = transformed['mask']
        
        # Ensure mask has channel dimension
        if isinstance(mask, torch.Tensor):
            if mask.dim() == 2:
                mask = mask.unsqueeze(0)
        else:
            mask = torch.tensor(mask, dtype=torch.float32).unsqueeze(0)
        
        return image, mask


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Train Wall Detection Model')
    parser.add_argument('--data-dir', type=str, default='data/cubicasa5k', help='Dataset directory')
    parser.add_argument('--output-dir', type=str, default='src/wall_detector/weights', help='Output directory')
    parser.add_argument('--image-size', type=int, default=512, help='Input image size')
    parser.add_argument('--batch-size', type=int, default=8, help='Batch size')
    parser.add_argument('--epochs', type=int, default=80, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=3e-4, help='Learning rate')
    parser.add_argument('--encoder', type=str, default='resnet34', help='Encoder backbone')
    parser.add_argument('--simple', action='store_true', help='Use simple threshold dataset')
    
    args = parser.parse_args()
    
    train(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        image_size=args.image_size,
        batch_size=args.batch_size,
        num_epochs=args.epochs,
        learning_rate=args.lr,
        use_simple_dataset=args.simple,
        encoder_name=args.encoder,
    )
