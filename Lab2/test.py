import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import argparse
import os
import numpy as np

# Import our custom model and dataset
from model import SnoutNetModel
from dataset import PetNoseDataset

# --- Define Paths ---
BASE_DATA_DIR = 'oxford-iiit-pet-noses'
IMG_DIR = os.path.join(BASE_DATA_DIR, 'images-original', 'images')
TEST_ANNOTATIONS = os.path.join(BASE_DATA_DIR, 'test_noses.txt')
# --------------------

def test(args):
    """Main testing loop to calculate accuracy statistics."""
    
    # 1. Setup Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 2. Create Test Dataset and DataLoader
    # As before, only ToTensor is needed. NO augmentation.
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
        num_workers=2
    )
    print(f"Loaded test dataset: {len(test_dataset)} samples")

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
    
    # Set model to evaluation mode
    model.eval()

    # 4. Testing Loop
    all_distances = []
    
    print("Running evaluation on test set...")
    with torch.no_grad(): # Disable gradient calculation
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            
            # Forward pass: get predictions
            predictions = model(images)
            
            # Calculate Euclidean distance for each sample in the batch
            # L2 distance = sqrt( (x1-x2)^2 + (y1-y2)^2 )
            distances = torch.sqrt(
                torch.sum((predictions - labels)**2, dim=1)
            )
            
            # Add batch distances to our master list
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
    # Setup command-line argument parsing
    parser = argparse.ArgumentParser(description='Test SnoutNet Model')
    
    parser.add_argument('--weights_path', type=str, required=True,
                        help='Path to the trained .pth model weights file')
    parser.add_widget('--batch_size', type=int, default=32,
                        help='Test batch size (default: 32)')
    
    args = parser.parse_args()
    
    test(args)
