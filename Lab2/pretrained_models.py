import torch
import torch.nn as nn
import torchvision.models as models

def SnoutNetAlexNet():
    """
    Loads a pretrained AlexNet and replaces the classifier
    for our 2-coordinate regression task.
    
    Following Step 2.6:
    1. Loads AlexNet with default (ImageNet) pretrained weights.
    2. Freezes all parameters in the 'features' (convolutional) layers.
    3. Replaces the 'classifier' with a new one suitable for regression.
    """
    
    # 1. Load pretrained model
    model = models.alexnet(weights=models.AlexNet_Weights.DEFAULT)
    
    # 2. Freeze all 'features' layers
    for param in model.features.parameters():
        param.requires_grad = False
        
    # 3. Replace the classifier
    # --- FIX ---
    # The original classifier's first linear layer is at index 1
    # It expects 9216 input features (256 * 6 * 6)
    num_features = model.classifier[1].in_features 
    # --- END FIX ---
    
    # Create a new classifier for regression
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.5),
        nn.Linear(num_features, 2048), # This will now be 9216 -> 2048
        nn.ReLU(inplace=True),
        nn.Dropout(p=0.5),
        nn.Linear(2048, 2048),
        nn.ReLU(inplace=True),
        nn.Linear(2048, 2)  # Output 2 values (x, y)
    )
    
    return model

def SnoutNetVGG():
    """
    Loads a pretrained VGG16 and replaces the classifier
    for our 2-coordinate regression task.
    
    Following Step 2.6:
    1. Loads VGG16 with default (ImageNet) pretrained weights.
    2. Freezes all parameters in the 'features' (convolutional) layers.
    3. Replaces the 'classifier' with a new one suitable for regression.
    """
    
    # 1. Load pretrained model
    model = models.vgg16(weights=models.VGG16_Weights.DEFAULT)

    # 2. Freeze all 'features' layers
    for param in model.features.parameters():
        param.requires_grad = False
        
    # 3. Replace the classifier
    # --- FIX ---
    # The original classifier's first linear layer is at index 0
    # It expects 25088 input features (512 * 7 * 7)
    num_features = model.classifier[0].in_features
    # --- END FIX ---
    
    # Create a new classifier for regression
    model.classifier = nn.Sequential(
        nn.Linear(num_features, 2048), # This will now be 25088 -> 2048
        nn.ReLU(inplace=True),
        nn.Dropout(p=0.5),
        nn.Linear(2048, 2048),
        nn.ReLU(inplace=True),
        nn.Linear(2048, 2)  # Output 2 values (x, y)
    )
    
    return model

if __name__ == "__main__":
    # You can run this file to test the models
    print("--- Testing AlexNet Finetune Model ---")
    alex_model = SnoutNetAlexNet()
    # print(alex_model)
    dummy_input = torch.randn(1, 3, 227, 227)
    output = alex_model(dummy_input)
    print(f"AlexNet output shape: {output.shape}") # Should be [1, 2]
    
    print("\n--- Testing VGG16 Finetune Model ---")
    vgg_model = SnoutNetVGG()
    # print(vgg_model)
    output = vgg_model(dummy_input)
    print(f"VGG16 output shape: {output.shape}") # Should be [1, 2]

