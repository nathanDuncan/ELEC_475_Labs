"""
SnoutNet Model for pet nose localization
Authour: Nathan Duncan
Assisted with: Google Gemini
"""
import torch
import torch.nn.functional as F
import torch.nn as nn

class SnoutNetModel(nn.Module):
    """
    Implementation of the SnoutNet model based on the architecture
    diagram in ELEC475_Lab2.pdf (Figure 1).
    
    This version uses 3x3 kernels as specified in the diagram's text,
    with padding and stride adjusted to match the output dimensions.
    """
    def __init__(self):
        super(SnoutNetModel, self).__init__()
        
        # --- Convolutional Blocks ---
        
        # Block 1: Input (B, 3, 227, 227) -> Output (B, 64, 57, 57)
        self.conv_block_1 = nn.Sequential(
            # --- MODIFIED ---
            # Using k=3, s=2, p=1 gets to (227-3+2*1)/2 + 1 = 114
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            # (114 - 2) / 2 + 1 = 57
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        # Block 2: Input (B, 64, 57, 57) -> Output (B, 128, 15, 15)
        self.conv_block_2 = nn.Sequential(
            # --- MODIFIED ---
            # Using k=3, s=2, p=2 gets to (57-3+2*2)/2 + 1 = 30
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=2, padding=2),
            nn.ReLU(),
            # --- MODIFIED ---
            # Using a standard MaxPool (30 - 2) / 2 + 1 = 15
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        # Block 3: Input (B, 128, 15, 15) -> Output (B, 256, 4, 4)
        self.conv_block_3 = nn.Sequential(
            # --- This block was already correct ---
            # (15 - 3 + 2*1) / 2 + 1 = 8
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            # (8 - 2) / 2 + 1 = 4
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        # --- Flattening ---
        # Flattens the (B, 256, 4, 4) output to (B, 256*4*4) or (B, 4096)
        self.flatten = nn.Flatten()
        
        # --- Fully Connected (Linear) Layers ---
        
        # FC1: Input (B, 4096) -> Output (B, 1024)
        self.fc_block_1 = nn.Sequential(
            nn.Linear(256 * 4 * 4, 1024),
            nn.ReLU()
        )
        
        # FC2: Input (B, 1024) -> Output (B, 1024)
        self.fc_block_2 = nn.Sequential(
            nn.Linear(1024, 1024),
            nn.ReLU()
        )
        
        # FC3 / Output: Input (B, 1024) -> Output (B, 2)
        # This is a regression, so no ReLU activation on the final output.
        self.fc_output = nn.Linear(1024, 2)

    def forward(self, x):
        """
        Defines the forward pass of the SnoutNet model.
        """
        # Pass through convolutional blocks
        x = self.conv_block_1(x)
        x = self.conv_block_2(x)
        x = self.conv_block_3(x)
        
        # Flatten for the fully connected layers
        x = self.flatten(x)
        
        # Pass through fully connected layers
        x = self.fc_block_1(x)
        x = self.fc_block_2(x)
        x = self.fc_output(x)
        
        return x
