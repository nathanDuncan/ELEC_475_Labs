import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import argparse
import os
import numpy as np
from tqdm import tqdm # <-- Import tqdm

# Import our custom model and dataset
from model import SnoutNetModel
from dataset import PetNoseDataset

# --- Define Paths ---
# We will now use the --data_dir argument
# BASE_DATA_DIR = 'oxford-iiit-pet-noses'
# IMG_DIR = os.path.join(BASE_DATA_DIR, 'images-original', 'images')
# TEST_ANNOTATIONS = os.path.join(BASE_DATA_DIR, 'test_noses.txt')
# --------------------

def test(args):
    """Main testing loop to calculate accuracy statistics."""
    
    # 1. Setup Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- UPDATED PATHS ---
    # Construct paths based on the --data_dir argument
    IMG_DIR = os.path.join(args.data_dir, 'images-original', 'images')
    TEST_ANNOTATIONS = os.path.join(args.data_dir, 'test_noses.txt')

    if not os.path.isdir(args.data_dir):
        print(f"Error: Data directory not found at {args.data_dir}")
        print("Please check the --data_dir path.")
        return
    # ---------------------

    # 2. Create Test Dataset and DataLoader
    test_transform = transforms.ToTensor()
    
    test_dataset = PetNoseDataset(
        annotations_file=TEST_ANNOTATIONS,
        img_dir=IMG_DIR,
        transform=test_transform,
        use_augmentation=False 
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers # <-- Use arg
    )
    print(f"Loaded test dataset: {len(test_dataset)} samples from {args.data_dir}")

    # 3. Initialize Model and Load Weights
    model = SnoutNetModel().to(device)
    
    try:
        model.load_state_dict(torch.load(args.weights_path, map_location=device))
    except FileNotFoundError:
        print(f"ERROR: Weights file not found at {args.weights_path}")
        print("Please train the model first using train.py")
        return
    except Exception as e:
        print(f"Error loading model weights: {e}")
        return
        
    print(f"Successfully loaded weights from {args.weights_path}")
    
    model.eval()

    # 4. Testing Loop
    all_distances = []
    
    print("Running evaluation on test set...")
    # Wrap test_loader in tqdm for a progress bar
    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="Testing"):
            images = images.to(device)
            labels = labels.to(device)
            
            predictions = model(images)
            
            distances = torch.sqrt(
                torch.sum((predictions - labels)**2, dim=1)
            )
            
            all_distances.extend(distances.cpu().numpy())

    # 5. Calculate and Print Statistics
    all_distances = np.array(all_distances)
    
    min_error = np.min(all_distances)
    max_error = np.max(all_distances)
    mean_error = np.mean(all_distances)
    std_error = np.std(all_distances)
    
    print("\n--- Test Results (Euclidean Distance in 227x227 space) ---")
    print(f"  Total Samples: {len(all_distances)}")
    print(f"  Mean Error:    {mean_error:.4f} pixels")
    print(f"  Std Deviation: {std_error:.4f} pixels")
    print(f"  Min Error:     {min_error:.4f} pixels")
    print(f"  Max Error:     {max_error:.4f} pixels")
    print("---------------------------------------------------------")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Test SnoutNet Model')
    
    # --- ADDED THIS ARGUMENT ---
    parser.add_argument('--data_dir', type=str, default='oxford-iiit-pet-noses',
                        help='Path to the base data directory')
    # ---------------------------

    parser.add_argument('--weights_path', type=str, required=True,
                        help='Path to the trained .pth model weights file')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Test batch size (default: 32)')
    parser.add_argument('--num_workers', type=int, default=0,
                        help='DataLoader num_workers. (default: 0, which is safest for Colab)')
    
    args = parser.parse_args()
    
    test(args)

