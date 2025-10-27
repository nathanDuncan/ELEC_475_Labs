import torch
import torch.nn as nn
import torchvision.transforms as transforms

# Import all the model architectures
from model import SnoutNetModel
from pretrained_models import SnoutNetAlexNet, SnoutNetVGG

class SnoutNetEnsemble(nn.Module):
    """
    Implements Step 2.7: An ensemble model that combines the
    predictions from SnoutNet, SnoutNetAlexNet, and SnoutNetVGG.
    
    It loads the three models with their trained weights and 
    averages their predictions during the forward pass.
    """
    
    def __init__(self, weights_snoutnet, weights_alexnet, weights_vgg, device):
        """
        Initializes the ensemble model.
        
        Args:
            weights_snoutnet (str): Path to the .pth weights for SnoutNetModel
            weights_alexnet (str): Path to the .pth weights for SnoutNetAlexNet
            weights_vgg (str): Path to the .pth weights for SnoutNetVGG
            device (torch.device): The device (cpu or cuda) to load models onto.
        """
        super(SnoutNetEnsemble, self).__init__()
        
        self.device = device
        
        # --- Define Normalization for Pretrained Models ---
        # The ensemble model must handle its own normalization,
        # as SnoutNet needs a raw tensor and the others need a normalized one.
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                              std=[0.229, 0.224, 0.225])

        # --- Load Model 1: SnoutNet ---
        self.model_snoutnet = SnoutNetModel().to(device)
        self.load_weights(self.model_snoutnet, weights_snoutnet, "SnoutNet")
        
        # --- Load Model 2: AlexNet ---
        self.model_alexnet = SnoutNetAlexNet().to(device)
        self.load_weights(self.model_alexnet, weights_alexnet, "SnoutNetAlexNet")

        # --- Load Model 3: VGG16 ---
        self.model_vgg = SnoutNetVGG().to(device)
        self.load_weights(self.model_vgg, weights_vgg, "SnoutNetVGG")

        # Set all models to evaluation mode
        self.model_snoutnet.eval()
        self.model_alexnet.eval()
        self.model_vgg.eval()

    def load_weights(self, model, weights_path, model_name):
        """Helper to load weights and freeze a model."""
        try:
            model.load_state_dict(torch.load(weights_path, map_location=self.device))
            print(f"Successfully loaded weights for {model_name} from {weights_path}")
        except FileNotFoundError:
            print(f"ERROR: Weights file not found for {model_name} at: {weights_path}")
            raise
        except Exception as e:
            print(f"Error loading weights for {model_name}: {e}")
            raise
            
        # Freeze all parameters
        for param in model.parameters():
            param.requires_grad = False

    def forward(self, x):
        """
        Forward pass for the ensemble.
        
        Args:
            x (torch.Tensor): The input image tensor (un-normalized).
        """
        
        # 1. Get prediction from SnoutNet (uses raw tensor)
        pred_snoutnet = self.model_snoutnet(x)
        
        # 2. Normalize the input for the pretrained models
        x_norm = self.normalize(x)
        
        # 3. Get predictions from AlexNet and VGG
        pred_alexnet = self.model_alexnet(x_norm)
        pred_vgg = self.model_vgg(x_norm)
        
        # 4. Average the predictions
        # Stack predictions: [ (1, 2), (1, 2), (1, 2) ] -> (3, B, 2)
        # We assume batch size B=1 during eval, but handle B > 1
        predictions_stack = torch.stack([pred_snoutnet, pred_alexnet, pred_vgg], dim=0)
        
        # Compute the mean along the 0th dimension (the models)
        final_prediction = torch.mean(predictions_stack, dim=0)
        
        return final_prediction

if __name__ == "__main__":
    # A simple test to check if the ensemble model can be created
    # This will fail if you don't have these exact weight files
    print("--- Testing Ensemble Model Creation ---")
    
    # Create dummy weight files for testing
    dummy_weights = "dummy_weights.pth"
    dummy_model = SnoutNetModel()
    torch.save(dummy_model.state_dict(), dummy_weights)

    try:
        device = torch.device("cpu")
        ensemble = SnoutNetEnsemble(
            weights_snoutnet=dummy_weights,
            weights_alexnet=dummy_weights, # This will fail (wrong keys), but tests logic
            weights_vgg=dummy_weights,     # This will fail (wrong keys), but tests logic
            device=device
        )
        print("Ensemble model created (with dummy weights).")
    except Exception as e:
        print(f"Caught expected error with dummy weights: {e}")
        
    # Clean up
    import os
    os.remove(dummy_weights)