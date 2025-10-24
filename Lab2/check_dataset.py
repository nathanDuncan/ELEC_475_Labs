import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import os

# (Added this import at the top)
import matplotlib
# Use a backend that doesn't require a GUI
matplotlib.use('Agg') 

# Attempt to import the PetNoseDataset from dataset.py
try:
    from dataset import PetNoseDataset
except ImportError:
    print("\n---")
    print("ERROR: Could not import 'PetNoseDataset' from 'dataset.py'.")
    print("Please make sure 'dataset.py' is in the same directory as 'check_dataset.py'.")
    print("---")
    exit()

# --- PATHS ---
BASE_DATA_DIR = 'oxford-iiit-pet-noses'
IMG_DIR = os.path.join(BASE_DATA_DIR, 'images-original', 'images')
ANNOTATIONS_FILE = os.path.join(BASE_DATA_DIR, 'train_noses.txt')
# -------------------------------------

BATCH_SIZE = 4 
OUTPUT_FILENAME = 'dataset_check.png' # We will save the plot here

def visualize_batch(images, labels):
    """
    Visualizes a batch of images and plots their corresponding
    ground-truth nose coordinates as a red cross.
    """
    print("Visualizing batch...")
    
    fig, axes = plt.subplots(1, BATCH_SIZE, figsize=(15, 5))
    if BATCH_SIZE == 1:
        axes = [axes]
        
    for i in range(BATCH_SIZE):
        image_tensor = images[i]
        coords = labels[i]
        
        img = image_tensor.permute(1, 2, 0)
        x, y = coords[0].item(), coords[1].item()
        
        axes[i].imshow(img)
        axes[i].plot(x, y, 'r+', markersize=12, markeredgewidth=2) 
        axes[i].set_title(f"Label: ({x:.1f}, {y:.1f})")
        axes[i].axis('off') 
        
    plt.suptitle("Dataset Reality Check: Images with GT Nose Coords", fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    # --- THIS IS THE KEY CHANGE ---
    # Instead of plt.show(), we save the figure to a file
    plt.savefig(OUTPUT_FILENAME)
    # --------------------------------
    
    print(f"\n--- SUCCESSFULLY SAVED VISUALS ---")
    print(f"Plot saved to: {OUTPUT_FILENAME}")
    print("Please refresh your file browser (folder icon on the left)")
    print(f"and double-click '{OUTPUT_FILENAME}' to view the image.")
    print(f"------------------------------------")


def run_reality_check():
    """
    Main function to run the dataset check.
    """
    print("--- Starting DataLoader Reality Check ---")
    
    if not os.path.isdir(IMG_DIR):
        print(f"ERROR: Image directory not found at: {IMG_DIR}")
        return
        
    if not os.path.isfile(ANNOTATIONS_FILE):
        print(f"ERROR: Annotations file not found at: {ANNOTATIONS_FILE}")
        return
    
    print(f"Found image directory: {IMG_DIR}")
    print(f"Found annotations file: {ANNOTATIONS_FILE}")
    
    data_transform = transforms.Compose([
        transforms.Resize((227, 227)),
        transforms.ToTensor(),
    ])
    
    print("Defined transforms (Resize to 227x227, ToTensor)")
    
    try:
        pet_dataset = PetNoseDataset(
            annotations_file=ANNOTATIONS_FILE,
            img_dir=IMG_DIR,
            transform=data_transform
        )
    except Exception as e:
        print(f"\nERROR: Failed to initialize PetNoseDataset.")
        print(f"Underlying error: {e}")
        return

    if len(pet_dataset) == 0:
        print("\nERROR: Dataset was initialized but loaded 0 samples.")
        return

    pet_loader = DataLoader(
        pet_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True, 
        num_workers=2
    )
    
    print(f"Initialized Dataset ({len(pet_dataset)} samples) and DataLoader (batch_size={BATCH_SIZE})")

    try:
        images, labels = next(iter(pet_loader))
        
        print("\n--- Batch Loaded Successfully ---")
        print(f"Image batch shape: {images.shape}")
        print(f"Labels batch shape: {labels.shape}")
        print("---------------------------------")
        
        visualize_batch(images, labels)
        
    except StopIteration:
        print("\nERROR: DataLoader is empty.")
    except Exception as e:
        print(f"\nAn error occurred while loading or visualizing the batch: {e}")


if __name__ == "__main__":
    run_reality_check()