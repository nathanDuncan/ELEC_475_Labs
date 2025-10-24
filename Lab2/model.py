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
    """
    def __init__(self):
        super(SnoutNetModel, self).__init__()
        
        # --- Convolutional Blocks ---
        # The architecture is interpreted as a series of (Conv2d -> ReLU -> MaxPool2d)
        # Parameters are chosen to match the feature map dimensions in the diagram.
        
        # Block 1: Input (B, 3, 227, 227) -> Output (B, 64, 57, 57)
        self.conv_block_1 = nn.Sequential(
            # (227 - 7 + 2*3) / 2 + 1 = 114
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7, stride=2, padding=3),
            nn.ReLU(),
            # (114 - 2 + 2*0) / 2 + 1 = 57
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        # Block 2: Input (B, 64, 57, 57) -> Output (B, 128, 15, 15)
        self.conv_block_2 = nn.Sequential(
            # (57 - 5 + 2*2) / 2 + 1 = 29
            # Note: A non-standard MaxPool is needed to match the diagram's 15x15 output
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            # (29 - 15 + 2*0) / 1 + 1 = 15
            nn.MaxPool2d(kernel_size=15, stride=1)
        )
        
        # Block 3: Input (B, 128, 15, 15) -> Output (B, 256, 4, 4)
        self.conv_block_3 = nn.Sequential(
            # (15 - 3 + 2*1) / 2 + 1 = 8
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            # (8 - 2 + 2*0) / 2 + 1 = 4
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
