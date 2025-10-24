"""
Test script for dimensionality of the SnoutNet model
Authour: Nathan Duncan
Assisted with: Google Gemini
"""
import torch
from model import SnoutNetModel

def test_model_architecture():
    """
    Performs the test described in Step 2.1 of the lab document.
    - Creates a dummy input tensor
    - Passes it through the model
    - Prints the output tensor's shape
    """
    print("--- SnoutNet Model Test ---")
    
    # 1. Initialize the model
    model = SnoutNetModel()
    model.eval()  # Set model to evaluation mode
    print("Model initialized.")

    # 2. Prepare a random dummy input tensor
    # Shape: BxCxWxH = 1 x 3 x 227 x 227
    try:
        dummy_input = torch.randn(1, 3, 227, 227)
        print(f"Created dummy input tensor with shape: {dummy_input.shape}")

        # 3. Forward pass
        with torch.no_grad():  # No need to track gradients for this test
            output = model(dummy_input)
        
        # 4. Check the output shape
        print(f"Model output shape: {output.shape}")
        
        expected_shape = torch.Size([1, 2])
        if output.shape == expected_shape:
            print(f"SUCCESS: Output shape ({output.shape}) matches expected shape ({expected_shape}).")
        else:
            print(f"FAILURE: Output shape ({output.shape}) does NOT match expected shape ({expected_shape}).")

    except Exception as e:
        print(f"\nAn error occurred during the model test: {e}")
        print("Please double-check the layer dimensions in 'model.py'.")

if __name__ == "__main__":
    test_model_architecture()