import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

# Attempt to import the PetNoseDataset from dataset.py
try:
    from dataset import PetNoseDataset
except ImportError:
    print("\n---")
    print("ERROR: Could not import 'PetNoseDataset' from 'dataset.py'.")
    print("Please make sure 'dataset.py' is in the same directory as 'check_dataset.py'.")
    print("---")
    exit()

# --- IMPORTANT: UPDATE THESE PATHS ---
# Update these paths to point to your unzipped dataset in Google Colab
# Example: '/content/pet_data/images-original/images'
IMG_DIR = '/content/pet_data/images-original/images'

# Example: '/content/pet_data/train-noses.txt'
ANNOTATIONS_FILE = '/content/pet_data/train-noses.txt'
# -------------------------------------

BATCH_SIZE = 4 # You can change this

def visualize_batch(images, labels):
    """
    Visualizes a batch of images and plots their corresponding
    ground-truth nose coordinates as a red cross.
    """
    print("Visualizing batch... (If images don't show, check your runtime environment)")
    
    # Create a grid of subplots
    fig, axes = plt.subplots(1, BATCH_SIZE, figsize=(15, 5))
    
    # Ensure axes is always an array, even for BATCH_SIZE=1
    if BATCH_SIZE == 1:
        axes = [axes]
        
    for i in range(BATCH_SIZE):
        # Get the i-th image and label from the batch
        image_tensor = images[i]
        coords = labels[i]
        
        # --- Convert Tensor to Plottable Image ---
        # 1. Permute from (C, H, W) to (H, W, C) for matplotlib
        img = image_tensor.permute(1, 2, 0)
        
        # 2. ToTensor() scales images to [0, 1]. No un-normalization
        #    is needed if we didn't apply transforms.Normalize
        
        # Get scaled coordinates
        x, y = coords[0].item(), coords[1].item()
        
        # Display the image
        axes[i].imshow(img)
        
        # Plot the red cross at the (x, y) coordinate
        axes[i].plot(x, y, 'r+', markersize=12, markeredgewidth=2) # Red cross
        
        axes[i].set_title(f"Label: ({x:.1f}, {y:.1f})")
        axes[i].axis('off') # Hide axes ticks
        
    plt.suptitle("Dataset Reality Check: Images with GT Nose Coords", fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    # In Colab, plt.show() should display the plot directly.
    plt.show()

def run_reality_check():
    """
    Main function to run the dataset check.
    """
    print("--- Starting DataLoader Reality Check ---")
    
    # --- 1. Define Transformations ---
    # As per the lab doc, SnoutNet requires 227x227 inputs.
    # ToTensor() converts PIL Image [0, 255] to torch.Tensor [0, 1]
    # and changes shape from (H, W, C) to (C, H, W).
    data_transform = transforms.Compose([
        transforms.Resize((227, 227)),
        transforms.ToTensor(),
        # We will add transforms.Normalize later in the training script
        # as it makes visualization harder here.
    ])
    
    print("Defined transforms (Resize to 227x227, ToTensor)")
    
    # --- 2. Initialize Dataset ---
    try:
        pet_dataset = PetNoseDataset(
            annotations_file=ANNOTATIONS_FILE,
            img_dir=IMG_DIR,
            transform=data_transform
        )
    except Exception as e:
        print(f"\nERROR: Failed to initialize PetNoseDataset.")
        print("Please check that your paths in check_dataset.py are correct.")
        print(f"IMG_DIR = '{IMG_DIR}'")
        print(f"ANNOTATIONS_FILE = '{ANNOTATIONS_FILE}'")
        print(f"Underlying error: {e}")
        return

    if len(pet_dataset) == 0:
        print("\nERROR: Dataset was initialized but loaded 0 samples.")
        print("Please check your annotation file and paths.")
        return

    # --- 3. Initialize DataLoader ---
    # DataLoader handles batching, shuffling, and parallel loading
    pet_loader = DataLoader(
        pet_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True, # Shuffle the data for good measure
        num_workers=2  # Use 2 workers to load data in parallel
    )
    
    print(f"Initialized Dataset ({len(pet_dataset)} samples) and DataLoader (batch_size={BATCH_SIZE})")

    # --- 4. Load One Batch ---
    try:
        # Get the first batch of data
        images, labels = next(iter(pet_loader))
        
        print("\n--- Batch Loaded Successfully ---")
        
        # Print shape and type information
        print(f"Image batch shape: {images.shape}")
        print(f"Image batch data type: {images.dtype}")
        
        print(f"Labels batch shape: {labels.shape}")
        print(f"Labels batch data type: {labels.dtype}")
        
        # Print example labels
        print("\nExample Labels (scaled to 227x227):")
        for i in range(BATCH_SIZE):
            print(f"  Sample {i}: [x={labels[i][0].item():.2f}, y={labels[i][1].item():.2f}]")
            
        print("---------------------------------")
        
        # --- 5. Visualize Batch ---
        visualize_batch(images, labels)
        
    except StopIteration:
        print("\nERROR: DataLoader is empty. This shouldn't happen unless the dataset is empty.")
    except Exception as e:
        print(f"\nAn error occurred while loading or visualizing the batch: {e}")
        print("Make sure matplotlib is installed and your paths are correct.")

if __name__ == "__main":
    # --- Check for placeholder paths ---
    if '/content/pet_data' in IMG_DIR or '/content/pet_data' in ANNOTATIONS_FILE:
        print("************************************************************")
        print("WARNING: You are using the default placeholder paths.")
        print("Please update 'IMG_DIR' and 'ANNOTATIONS_FILE' variables")
        print("in 'check_dataset.py' to point to your dataset location.")
        print("************************************************************")
    
    run_reality_check()