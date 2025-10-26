import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import argparse
import os
import time
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm # <-- Import tqdm

# Import our custom model and dataset
from model import SnoutNetModel
from dataset import PetNoseDataset

# --- Define Paths ---
# We will now use the --data_dir argument to define the base path
# BASE_DATA_DIR = 'oxford-iiit-pet-noses' # <-- This is now an argument
# IMG_DIR = os.path.join(BASE_DATA_DIR, 'images-original', 'images')
# TRAIN_ANNOTATIONS = os.path.join(BASE_DATA_DIR, 'train_noses.txt')
# TEST_ANNOTATIONS = os.path.join(BASE_DATA_DIR, 'test_noses.txt')
# --------------------

def train(args):
    """Main training and validation loop."""
    
    # 1. Setup Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- UPDATED PATHS ---
    # Construct paths based on the --data_dir argument
    IMG_DIR = os.path.join(args.data_dir, 'images-original', 'images')
    TRAIN_ANNOTATIONS = os.path.join(args.data_dir, 'train_noses.txt')
    TEST_ANNOTATIONS = os.path.join(args.data_dir, 'test_noses.txt')
    
    if not os.path.isdir(args.data_dir):
        print(f"Error: Data directory not found at {args.data_dir}")
        print("Please check the --data_dir path.")
        return
    # ---------------------

    # 2. Create Datasets and DataLoaders
    data_transform = transforms.ToTensor()

    # --- Training Dataset ---
    train_dataset = PetNoseDataset(
        annotations_file=TRAIN_ANNOTATIONS,
        img_dir=IMG_DIR,
        transform=data_transform,
        use_augmentation=args.use_augmentation
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers # <-- Use arg
    )
    print(f"Loaded training dataset: {len(train_dataset)} samples from {args.data_dir}")
    aug_status = "ENABLED" if args.use_augmentation else "DISABLED"
    print(f"Augmentation: {aug_status}")

    # --- Validation Dataset ---
    val_dataset = PetNoseDataset(
        annotations_file=TEST_ANNOTATIONS, # Use test set for validation
        img_dir=IMG_DIR,
        transform=data_transform,
        use_augmentation=False 
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False, 
        num_workers=args.num_workers # <-- Use arg
    )
    print(f"Loaded validation dataset: {len(val_dataset)} samples")

    # 3. Initialize Model, Loss, and Optimizer
    model = SnoutNetModel().to(device)
    
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    # 4. Training Loop
    best_val_loss = float('inf')
    history = {'epoch': [], 'train_loss': [], 'val_loss': []}

    print(f"\n--- Starting Training for {args.epochs} epochs ---")
    start_time = time.time()

    for epoch in range(args.epochs):
        
        # --- Training Phase ---
        model.train()
        running_train_loss = 0.0
        # Wrap train_loader in tqdm for a progress bar
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs} [Train]")
        
        for images, labels in train_pbar:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_train_loss += loss.item() * images.size(0)
            
            # Update the progress bar description with the current loss
            train_pbar.set_postfix({'loss': loss.item()})

        epoch_train_loss = running_train_loss / len(train_loader.dataset)

        # --- Validation Phase ---
        model.eval()
        running_val_loss = 0.0
        # Wrap val_loader in tqdm for a progress bar
        val_pbar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{args.epochs} [Val]")
        
        with torch.no_grad():
            for images, labels in val_pbar:
                images = images.to(device)
                labels = labels.to(device)
                
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                running_val_loss += loss.item() * images.size(0)
                val_pbar.set_postfix({'val_loss': loss.item()})

        epoch_val_loss = running_val_loss / len(val_loader.dataset)
        
        # We can clear the progress bars now
        train_pbar.close()
        val_pbar.close()
        
        # Print epoch stats
        print(f"Epoch {epoch+1}/{args.epochs} | "
              f"Train Loss: {epoch_train_loss:.6f} | "
              f"Val Loss: {epoch_val_loss:.6f}")
        
        history['epoch'].append(epoch + 1)
        history['train_loss'].append(epoch_train_loss)
        history['val_loss'].append(epoch_val_loss)

        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            torch.save(model.state_dict(), args.weights_name)
            print(f"  -> New best model saved to {args.weights_name}")

    end_time = time.time()
    print(f"--- Training Finished ---")
    print(f"Total time: {(end_time - start_time) / 60:.2f} minutes")
    print(f"Best validation loss: {best_val_loss:.6f}")
    
    # 5. Save Loss Plot
    history_df = pd.DataFrame(history)
    plot_filename = f"loss_plot_{args.weights_name.replace('.pth', '.png')}"
    plt.figure(figsize=(10, 5))
    plt.plot(history_df['epoch'], history_df['train_loss'], label='Train Loss')
    plt.plot(history_df['epoch'], history_df['val_loss'], label='Validation Loss')
    plt.title(f"Training and Validation Loss (Augmentation: {aug_status})")
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(plot_filename)
    print(f"Loss plot saved to {plot_filename}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train SnoutNet Model')
    
    # --- ADDED THIS ARGUMENT ---
    parser.add_argument('--data_dir', type=str, default='oxford-iiit-pet-noses',
                        help='Path to the base data directory')
    # ---------------------------

    parser.add_argument('--use_augmentation', action='store_true',
                        help='Enable data augmentation (flip and jitter)')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate (default: 0.0001)')
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of training epochs (default: 50)')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Training batch size (default: 32)')
    parser.add_argument('--weights_name', type=str, default='snoutnet.pth',
                        help='Name to save the trained model weights')
    parser.add_argument('--num_workers', type=int, default=0,
                        help='DataLoader num_workers. (default: 0, which is safest for Colab)')
    
    args = parser.parse_args()
    
    train(args)
