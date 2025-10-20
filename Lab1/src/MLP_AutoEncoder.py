# Imports 
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
# import torch-summary as ts

#####################################################
def visualize_MNIST_img():
    """
    Write a program that loads the MNIST dataset, 
    prompts the user to input an integer index between 0 and 59999, 
    and displays the indexed image and its label.
    """
    ### READ INPUT ###
    idx = input("Please enter the index (0-59999) of the MNIST image you wish to visualize...")
    idx = int(idx)

    ### LOAD IMAGE ###
    train_transform = transforms.Compose([transforms.ToTensor()]) 
    train_set = datasets.MNIST('./data/mnist', train=True, download=True,
                      transform=train_transform) 

    ### PLOT IMAGE ###
    plt.imshow(train_set.data[idx], cmap='gray') 
    plt.show() 


if __name__ == "__main__":
    print("[INFO] Beginning Program...")

    #2. Visualizing the MNIST Dataset
    print("[INFO] Starting MNIST Visualizer...")
    visualize_MNIST_img()

    

    print("[INFO] Program finished.")
    #

