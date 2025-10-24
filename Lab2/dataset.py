import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import re

class PetNoseDataset(Dataset):
    """
    Custom PyTorch Dataset for the Oxford-IIIT Pet Nose Localization task.
    
    Reads image filenames and (x, y) coordinates from an annotation file,
    loads images, and applies necessary transformations.
    """
    
    def __init__(self, annotations_file, img_dir, transform=None):
        """
        Args:
            annotations_file (string): Path to the .txt file with annotations.
            img_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on an image sample.
        """
        self.img_dir = img_dir
        self.transform = transform
        self.annotations = []
        
        # Regex to safely parse the coordinate string, e.g., "(198, 304)"
        # It handles potential leading/trailing spaces.
        self.coord_regex = re.compile(r'\(\s*(\d+)\s*,\s*(\d+)\s*\)')
        
        print(f"Loading annotations from: {annotations_file}")
        
        try:
            with open(annotations_file, 'r') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                        
                    parts = line.split(',', 1) # Split only on the first comma
                    if len(parts) != 2:
                        print(f"Warning: Skipping malformed line: {line}")
                        continue
                        
                    img_name = parts[0].strip()
                    coord_str = parts[1].strip().strip('"') # Remove quotes
                    
                    match = self.coord_regex.match(coord_str)
                    if not match:
                        print(f"Warning: Skipping line with unparsable coords: {line}")
                        continue
                    
                    # Coords are read as (x, y)
                    x_orig = int(match.group(1))
                    y_orig = int(match.group(2))
                    
                    self.annotations.append((img_name, (x_orig, y_orig)))
                    
        except FileNotFoundError:
            print(f"ERROR: Annotation file not found at {annotations_file}")
            raise
        except Exception as e:
            print(f"ERROR: Failed to read annotations file: {e}")
            raise

        if not self.annotations:
            print(f"ERROR: No annotations were successfully loaded from {annotations_file}.")
            print("Please check the file path and format.")
        else:
            print(f"Successfully loaded {len(self.annotations)} annotations.")

    def __len__(self):
        """Returns the total number of samples in the dataset."""
        return len(self.annotations)

    def __getitem__(self, idx):
        """
        Fetches the sample at the given index.
        
        Returns:
            tuple: (image, label) where image is the transformed image
            tensor and label is a tensor of the scaled [x, y] coordinates.
        """
        if torch.is_tensor(idx):
            idx = idx.tolist()
            
        img_name, (x_orig, y_orig) = self.annotations[idx]
        img_path = os.path.join(self.img_dir, img_name)
        
        try:
            # Open image and ensure it's in RGB format
            image = Image.open(img_path).convert('RGB')
        except FileNotFoundError:
            print(f"ERROR: Image file not found at {img_path} (for annotation: {img_name})")
            # Return a placeholder or skip? For now, re-raise.
            raise
        except Exception as e:
            print(f"ERROR: Could not open image {img_path}: {e}")
            raise
            
        # Get original image dimensions
        w_orig, h_orig = image.size
        
        # --- Coordinate Scaling ---
        # Scale the original coordinates to the target size (227x227)
        # as required by the lab document.
        # (x_new) = (x_orig) * (target_width / orig_width)
        # (y_new) = (y_orig) * (target_height / orig_height)
        
        target_size = 227.0 # Use float for precision
        x_scaled = x_orig * (target_size / w_orig)
        y_scaled = y_orig * (target_size / h_orig)
        
        # Create the label tensor
        label = torch.tensor([x_scaled, y_scaled], dtype=torch.float32)

        # Apply transformations to the image
        # The transform pipeline MUST include transforms.Resize((227, 227))
        # and transforms.ToTensor()
        if self.transform:
            image = self.transform(image)
            
        return image, label
