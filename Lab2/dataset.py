import torch
import pandas as pd
from PIL import Image
import os
import ast
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF # For H-Flip

class PetNoseDataset(Dataset):
    """
    Custom Dataset for the Pet Nose Localization task.
    
    Handles loading images, parsing coordinates, and applying
    transforms and augmentations.
    """
    def __init__(self, annotations_file, img_dir, 
                 transform=None, use_augmentation=False):
        """
        Args:
            annotations_file (string): Path to the csv file with annotations.
            img_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample (e.g., ToTensor()).
            use_augmentation (bool): If True, applies random augmentations.
        """
        # --- FIX 1 ---
        # The annotation file only has 2 columns
        self.annotations_df = pd.read_csv(
            annotations_file, 
            header=None, 
            names=['filename', 'coords_str'] 
        )
        self.img_dir = img_dir
        self.transform = transform
        self.use_augmentation = use_augmentation
        
        # Define our standard and augmentation transforms
        self.resize = transforms.Resize((227, 227))
        self.aug_color = transforms.ColorJitter(brightness=0.3, 
                                                contrast=0.3, 
                                                saturation=0.3)
        self.to_tensor = transforms.ToTensor()

    def __len__(self):
        return len(self.annotations_df)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # 1. Load image path and metadata
        row = self.annotations_df.iloc[idx]
        img_name = os.path.join(self.img_dir, row['filename'])
        image = Image.open(img_name).convert('RGB')
        
        # --- FIX 2 ---
        # Get the original width and height *from the image itself*
        # before resizing it.
        original_w, original_h = image.size
        
        # 2. Parse coordinates from the string
        # Use ast.literal_eval for safe parsing of "(x, y)" string
        coords_tuple = ast.literal_eval(row['coords_str'].strip())
        
        # 3. Apply base resize to image
        image = self.resize(image)
        
        # 4. Scale coordinates to the resized 227x227 space
        # This math will now work correctly
        scaled_x = coords_tuple[0] * (227.0 / original_w)
        scaled_y = coords_tuple[1] * (227.0 / original_h)
        coords = torch.tensor([scaled_x, scaled_y], dtype=torch.float32)

        # 5. Apply Augmentations (if enabled)
        if self.use_augmentation:
            
            # --- Augmentation 1: Horizontal Flip ---
            # 50% chance to flip
            if torch.rand(1) < 0.5:
                image = TF.hflip(image)
                # IMPORTANT: We must also flip the x-coordinate
                # A 0-indexed pixel at x becomes (width - 1 - x)
                coords[0] = 227.0 - 1.0 - coords[0]
                
            # --- Augmentation 2: Color Jitter ---
            # This transform doesn't affect coordinates
            image = self.aug_color(image)

        # 6. Apply final transform (e.g., ToTensor)
        # We must apply ToTensor *after* all PIL-based augmentations
        if self.transform:
            image_tensor = self.transform(image)
        else:
            # Fallback if no transform is provided
            image_tensor = self.to_tensor(image)

        return image_tensor, coords