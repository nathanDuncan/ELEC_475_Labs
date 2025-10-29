import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import argparse
import os
import numpy as np
from tqdm import tqdm 
import time
import matplotlib
# Use a backend that doesn't require a GUI
matplotlib.use('Agg') 
import matplotlib.pyplot as plt

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

# --- NEW HELPER FUNCTIONS ---

def unnormalize(tensor, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    """Reverses ImageNet normalization for plotting."""
    # Clone to avoid in-place modification
    tensor = tensor.clone() 
    mean = torch.tensor(mean, device=tensor.device).view(3, 1, 1)
    std = torch.tensor(std, device=tensor.device).view(3, 1, 1)
    tensor.mul_(std).add_(mean)
    return tensor

def visualize_results(image_list, label_list, pred_list, title_prefix, filename, is_normalized):
    """
    Saves a 2x4 plot of the 4 best and 4 worst predictions.
    Plots Ground Truth (green +) and Prediction (red x).
    """
    print(f"Generating visualizations... saving to {filename}")
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    
    for i in range(8):
        row = i // 4
        col = i % 4
        
        if row == 0:
            ax_title = f"Best #{col+1}"
        else:
            ax_title = f"Worst #{col+1}"
            
        ax = axes[row, col]
        
        # Get data
        img_tensor = image_list[i]
        gt_coords = label_list[i]
        pred_coords = pred_list[i]
        
        # Un-normalize if needed
        if is_normalized:
            img_tensor = unnormalize(img_tensor)
            
        # Convert image to plottable format
        img = img_tensor.cpu().permute(1, 2, 0).numpy()
        img = np.clip(img, 0, 1) # Ensure valid pixel range
        
        # Get coords
        gt_x, gt_y = gt_coords[0].item(), gt_coords[1].item()
        pred_x, pred_y = pred_coords[0].item(), pred_coords[1].item()
        
        ax.imshow(img)
        # Plot Ground Truth (Green Cross)
        ax.plot(gt_x, gt_y, 'g+', markersize=12, markeredgewidth=2, label='Ground Truth') 
        # Plot Prediction (Red X)
        ax.plot(pred_x, pred_y, 'rx', markersize=12, markeredgewidth=2, label='Prediction')
        
        ax.set_title(ax_title)
        ax.axis('off')
        
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper right', fontsize=14)
    fig.suptitle(f"Prediction Visualization: {title_prefix}", fontsize=20)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(filename)
    print(f"Successfully saved plot to {filename}")

# --------------------------

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
    if not args.weights_path:
        raise ValueError(f"Must provide --weights_path for model type '{model_name}'")
        
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
    is_normalized = False # Flag for visualization
    if args.model == 'snoutnet' or args.model == 'ensemble':
        test_transform = transforms.ToTensor()
        if args.model == 'ensemble':
            print("Using basic ToTensor() transform (Ensemble handles normalization internally)")
        else:
            print("Using basic ToTensor() transform for SnoutNet")
    else:
        # alexnet or vgg16
        is_normalized = True
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
    # --- MODIFIED: Store all results for sorting ---
    all_results = [] # Stores (distance, image, label, prediction)
    total_inference_time = 0.0
    total_images = 0
    # ---------------------------------------------
    
    print("Running evaluation on test set...")
    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="Testing"):
            images = images.to(device)
            labels = labels.to(device)
            
            # --- Time the inference ---
            start_time = time.time()
            predictions = model(images)
            end_time = time.time()
            total_inference_time += (end_time - start_time)
            total_images += images.size(0)
            # --------------------------
            
            distances = torch.sqrt(
                torch.sum((predictions - labels)**2, dim=1)
            )
            
            # Store all results for later sorting
            for i in range(images.size(0)):
                all_results.append((
                    distances[i].item(),
                    images[i].cpu(),  # Store on CPU to save GPU memory
                    labels[i].cpu(),
                    predictions[i].cpu()
                ))

    # 5. Calculate and Print Statistics
    
    # --- Sort by distance (first element of tuple) ---
    all_results.sort(key=lambda x: x[0])
    
    # --- Extract sorted data ---
    all_distances = np.array([res[0] for res in all_results])
    
    best_4_results = all_results[:4]
    worst_4_results = all_results[-4:]
    
    best_4_distances = np.array([res[0] for res in best_4_results])
    worst_4_distances = np.array([res[0] for res in worst_4_results])
    
    # --- Overall Stats ---
    print(f"\n--- Test Results (Euclidean Distance in 227x227 space) ---")
    print(f"  Model: {args.model}")
    print(f"  Total Samples: {len(all_distances)}")
    print(f"  --- Overall ---")
    print(f"  Mean Error:    {np.mean(all_distances):.4f} pixels")
    print(f"  Std Deviation: {np.std(all_distances):.4f} pixels")
    print(f"  Min Error:     {np.min(all_distances):.4f} pixels")
    print(f"  Max Error:     {np.max(all_distances):.4f} pixels")
    
    # --- 4 Best Stats ---
    print(f"  --- 4 Best ---")
    print(f"  Mean Error:    {np.mean(best_4_distances):.4f} pixels")
    print(f"  Std Deviation: {np.std(best_4_distances):.4f} pixels")
    print(f"  Min Error:     {np.min(best_4_distances):.4f} pixels")
    print(f"  Max Error:     {np.max(best_4_distances):.4f} pixels")
    
    # --- 4 Worst Stats ---
    print(f"  --- 4 Worst ---")
    print(f"  Mean Error:    {np.mean(worst_4_distances):.4f} pixels")
    print(f"  Std Deviation: {np.std(worst_4_distances):.4f} pixels")
    print(f"  Min Error:     {np.min(worst_4_distances):.4f} pixels")
    print(f"  Max Error:     {np.max(worst_4_distances):.4f} pixels")
    
    # --- Time Performance ---
    avg_time_msec = (total_inference_time / total_images) * 1000
    print(f"  --- Performance ---")
    print(f"  Time/Image:    {avg_time_msec:.4f} msec")
    print("---------------------------------------------------------")
    
    # 6. Generate Visualizations if requested
    if args.visualize:
        vis_images = [res[1] for res in best_4_results] + [res[1] for res in worst_4_results]
        vis_labels = [res[2] for res in best_4_results] + [res[2] for res in worst_4_results]
        vis_preds = [res[3] for res in best_4_results] + [res[3] for res in worst_4_results]
        
        # Generate a dynamic filename
        if args.model == 'ensemble':
            vis_filename = f"visualize_ensemble.png"
        else:
            base_name = os.path.basename(args.weights_path).replace('.pth', '')
            vis_filename = f"visualize_{base_name}.png"
        
        visualize_results(
            vis_images, 
            vis_labels, 
            vis_preds, 
            title_prefix=f"Model: {args.model}", 
            filename=vis_filename,
            is_normalized=is_normalized
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Test SnoutNet Model')
    
    parser.add_argument('--model', type=str, default='snoutnet',
                        help="Model to test. Choose 'snoutnet', 'alexnet', 'vgg16', or 'ensemble'")
    parser.add_argument('--data_dir', type=str, default='oxford-iiit-pet-noses',
                        help='Path to the base data directory')
    
    # --- NEW: Argument to trigger visualization ---
    parser.add_argument('--visualize', action='store_true',
                        help='Generate and save visualization plots of best/worst 4.')

    # --- Args for individual models ---
    parser.add_argument('--weights_path', type=str, default=None,
                        help='Path to the trained .pth model weights file (for snoutnet, alexnet, vgg16)')

    # --- Args for ENSEMBLE model ---
    parser.add_argument('--weights_snoutnet', type=str, default=None,
                        help='(ENSEMBBLE ONLY) Path to trained SnoutNet weights')
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
