import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import argparse
import os
import numpy as np
from tqdm import tqdm 

# --- UPDATED IMPORTS ---
# Import all our custom models
from model import SnoutNetModel
try:
    from pretrained_models import SnoutNetAlexNet, SnoutNetVGG
    from ensemble_model import SnoutNetEnsemble
except ImportError as e:
    print(f"Warning: Could not import models: {e}")
    print("Only 'snoutnet' model type might be available.")
# -------------------------

from dataset import PetNoseDataset


def load_model(args, device):
    """Helper function to load the correct model architecture."""
    model_name = args.model
    
    if model_name == 'snoutnet':
        print("Loading SnoutNetModel")
        model = SnoutNetModel()
    elif model_name == 'alexnet':
        print("Loading SnoutNetAlexNet")
        model = SnoutNetAlexNet()
    elif model_name == 'vgg16':
        print("Loading SnoutNetVGG")
        model = SnoutNetVGG()
    elif model_name == 'ensemble':
        print("Loading SnoutNetEnsemble")
        if not (args.weights_snoutnet and args.weights_alexnet and args.weights_vgg):
            raise ValueError("For ensemble model, you must provide --weights_snoutnet, --weights_alexnet, and --weights_vgg")
        model = SnoutNetEnsemble(
            weights_snoutnet=args.weights_snoutnet,
            weights_alexnet=args.weights_alexnet,
            weights_vgg=args.weights_vgg,
            device=device
        )
        return model # Ensemble loads its own weights, so we return early
    else:
        raise ValueError(f"Unknown model name: {model_name}. Choose 'snoutnet', 'alexnet', 'vgg16', or 'ensemble'")
    
    # Load weights for non-ensemble models
    model = model.to(device)
    try:
        model.load_state_dict(torch.load(args.weights_path, map_location=device))
    except FileNotFoundError:
        print(f"ERROR: Weights file not found at {args.weights_path}")
        print("Please train the model first using train.py")
        raise
    except Exception as e:
        print(f"Error loading model weights: {e}")
        raise
        
    print(f"Successfully loaded weights from {args.weights_path}")
    return model


def test(args):
    """Main testing loop to calculate accuracy statistics."""
    
    # 1. Setup Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- UPDATED PATHS ---
    IMG_DIR = os.path.join(args.data_dir, 'images-original', 'images')
    TEST_ANNOTATIONS = os.path.join(args.data_dir, 'test_noses.txt')

    if not os.path.isdir(args.data_dir):
        print(f"Error: Data directory not found at {args.data_dir}")
        print("Please check the --data_dir path.")
        return
    # ---------------------

    # 2. Create Test Dataset and DataLoader
    # --- UPDATED TRANSFORMS ---
    # The ensemble model needs the raw ToTensor() output
    # Other models need normalization (except snoutnet)
    if args.model == 'snoutnet' or args.model == 'ensemble':
        test_transform = transforms.ToTensor()
        if args.model == 'ensemble':
            print("Using basic ToTensor() transform (Ensemble handles normalization internally)")
        else:
            print("Using basic ToTensor() transform for SnoutNet")
    else:
        # alexnet or vgg16
        test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
        print(f"Using ImageNet normalization transforms for {args.model}")
    # --------------------------
    
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
        num_workers=args.num_workers
    )
    print(f"Loaded test dataset: {len(test_dataset)} samples from {args.data_dir}")

    # 3. Initialize Model and Load Weights
    try:
        model = load_model(args, device)
    except NameError:
        print("ERROR: One or more models not found. Make sure all .py files are present.")
        return
    except ValueError as e:
        print(f"ERROR: {e}")
        return
    except Exception as e:
        print(f"An unexpected error occurred loading the model: {e}")
        return
    
    model.eval()

    # 4. Testing Loop
    all_distances = []
    
    print("Running evaluation on test set...")
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
    
    print(f"\n--- Test Results (Euclidean Distance in 227x227 space) ---")
    print(f"  Model: {args.model}")
    print(f"  Total Samples: {len(all_distances)}")
    print(f"  Mean Error:    {mean_error:.4f} pixels")
    print(f"  Std Deviation: {std_error:.4f} pixels")
    print(f"  Min Error:     {min_error:.4f} pixels")
    print(f"  Max Error:     {max_error:.4f} pixels")
    print("---------------------------------------------------------")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Test SnoutNet Model')
    
    parser.add_argument('--model', type=str, default='snoutnet',
                        help="Model to test. Choose 'snoutnet', 'alexnet', 'vgg16', or 'ensemble'")
    parser.add_argument('--data_dir', type=str, default='oxford-iiit-pet-noses',
                        help='Path to the base data directory')

    # --- Args for individual models ---
    parser.add_argument('--weights_path', type=str, default=None,
                        help='Path to the trained .pth model weights file (for snoutnet, alexnet, vgg16)')

    # --- Args for ENSEMBLE model ---
    parser.add_argument('--weights_snoutnet', type=str, default=None,
                        help='(ENSEMBLE ONLY) Path to trained SnoutNet weights')
    parser.add_argument('--weights_alexnet', type=str, default=None,
                        help='(ENSEMBLE ONLY) Path to trained AlexNet weights')
    parser.add_argument('--weights_vgg', type=str, default=None,
                        help='(ENSEMBLE ONLY) Path to trained VGG16 weights')

    parser.add_argument('--batch_size', type=int, default=32,
                        help='Test batch size (default: 32)')
    parser.add_argument('--num_workers', type=int, default=0,
                        help='DataLoader num_workers. (default: 0, which is safest for Colab)')
    
    args = parser.parse_args()
    
    test(args)

