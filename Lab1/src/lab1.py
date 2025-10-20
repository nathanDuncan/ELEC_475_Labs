
#########################################################################################################
#
#   ELEC 475 - Lab 1, Step 1
#   Fall 2025
#

import torch
import torchvision.transforms as transforms
import argparse
import matplotlib.pyplot as plt
import numpy as np
from torchvision.datasets import MNIST
from model import autoencoderMLP4Layer

PRINTS = False
MNIST_LOW = 0
MNIST_HIGH = 59999

##### STEP 2: VISUALIZING THE MNIST DATASET ##################################################
def visualize_MNIST_img(user_input):
    """
    Write a program that loads the MNIST dataset, 
    prompts the user to input an integer index between 0 and 59999, 
    and displays the indexed image and its label.
    """
    print("\n[INFO] Step 2: Starting MNIST Visualizer...")
    ### READ INPUT ###
    if user_input:
        idx = input(f"Please enter the index ({MNIST_LOW}-{MNIST_HIGH}) of the MNIST image you wish to visualize...\n")
        idx = int(idx)
    else:
        idx = 0

    ### LOAD IMAGE ###
    train_transform = transforms.Compose([transforms.ToTensor()]) 
    train_set = MNIST('./data/mnist', train=True, download=True,
                      transform=train_transform) 

    ### PLOT IMAGE ###
    plt.imshow(train_set.data[idx], cmap='gray') 
    plt.show() 

##### STEP 4/5: TEST YOUR AUTOENCODER ########################################################
def autoencoder_test(user_input=0, save_file=None, bottleneck_size=0, noise=False, sets=3):
    """
    Step 4
    Modify your visualization program from Step 2, to show the reconstruction results from the 
    autoencoder. First, instantiate a version of your model, with the MLP.8.pth network weights that you 
    generated during training. Then, pass each indexed image as input to the model, and display both the 
    input and the output images side-by-side.
    """
    """
    Step 5
    Autoencoders can be used to remove noise from an image. Test your autoencoder’s ability to remove 
    image noise, by repeating the test in Step 4 with noise added to each image.
    """
    if not noise: print('\n[INFO] Step 4: Testing autoencoder...')
    else: print('\n[INFO] Step 5: Testing image denoising...')

    device = 'cpu'
    if torch.cuda.is_available():
        device = 'cuda'
    print('[INFO] Using device ', device)

    train_transform = transforms.Compose([
        transforms.ToTensor()
    ])
    test_transform = train_transform

    train_set = MNIST('./data/mnist', train=True, download=True, transform=train_transform)
    test_set = MNIST('./data/mnist', train=False, download=True, transform=test_transform)
    # train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)
    # test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False)

    N_input = 28 * 28   # MNIST image size
    N_output = N_input
    model = autoencoderMLP4Layer(N_input=N_input, N_bottleneck=bottleneck_size, N_output=N_output)
    model.load_state_dict(torch.load(save_file))
    model.to(device)
    model.eval()

    idx = 0
    while idx >= 0:
        if user_input:
            idx = input("Enter index > ")
            idx = int(idx)
        else: 
            if idx: break
            idx = 1
        imgs = []
        for k in range(sets):
            img_set = []
            if 0 <= idx <= train_set.data.size()[0]:
                if PRINTS: print('label = ', train_set.targets[idx].item())
                img = train_set.data[idx]
                if PRINTS: print('break 9', img.shape, img.dtype, torch.min(img), torch.max(img))

                img = img.type(torch.float32)
                if PRINTS: print('break 10', img.shape, img.dtype, torch.min(img), torch.max(img))
                img = (img - torch.min(img)) / torch.max(img)
                if PRINTS: print('break 11', img.shape, img.dtype, torch.min(img), torch.max(img))

                # plt.imshow(img, cmap='gray')
                # plt.show()

                img = img.to(device=device)
                # print('break 7: ', torch.max(img), torch.min(img), torch.mean(img))
                if PRINTS: print('break 8 : ', img.shape, img.dtype)
                img = img.view(1, img.shape[0]*img.shape[1]).type(torch.FloatTensor)
                if PRINTS: print('break 9 : ', img.shape, img.dtype)

                # Add noise to test if desired
                if noise: test_img = img + torch.rand_like(img)
                else: test_img = img 

                with torch.no_grad():
                    output = model(test_img)
                # output = output.view(28, 28).type(torch.ByteTensor)
                # output = output.view(28, 28).type(torch.FloatTensor)
                output = output.view(28, 28).type(torch.FloatTensor)
                if PRINTS: print('break 10 : ', output.shape, output.dtype)
                if PRINTS: print('break 11: ', torch.max(output), torch.min(output), torch.mean(output))
                # plt.imshow(output, cmap='gray')
                # plt.show()

                # both = np.hstack((img.view(28, 28).type(torch.FloatTensor),output))
                # plt.imshow(both, cmap='gray')
                # plt.show()
            
                idx += 1

                img = img.view(28, 28).type(torch.FloatTensor)
                test_img = test_img.view(28, 28).type(torch.FloatTensor)

                if noise: img_set = [img, test_img, output]
                else: img_set = [test_img, output]

            if img_set: imgs.append(img_set)

        # plot results if there was a list created
        if imgs: plot(imgs)

##### STEP 6: BOTTLENECK INTERPOLATION #######################################################
def bottleneck_interpolation(user_input=0, save_file=None, sets=3, interpolation_steps=8):
    """
    module that passes two images through the encode method, returning their two 
    bottleneck tensors. Then, linearly interpolate through these two tensors for n steps, creating a set of n
    new bottleneck tensors. Pass each of these new tensors through the decode method, and plot the results
    """
    print('\n[INFO] Step 6: Testing interpolation...')

    device = 'cpu'
    if torch.cuda.is_available():
        device = 'cuda'
    print('[INFO] Using device ', device)

    N_input = 28 * 28   # MNIST image size
    N_output = N_input
    model = autoencoderMLP4Layer(N_input=N_input, N_bottleneck=8, N_output=N_output)
    model.load_state_dict(torch.load(save_file))
    model.to(device)
    model.eval()

    train_transform = transforms.Compose([
        transforms.ToTensor()
    ])
    train_set = MNIST('./data/mnist', train=True, download=True, transform=train_transform)

    imgs = []

    for i in range(sets):
        # Select the two base images
        test_indexes = torch.randint(MNIST_LOW, MNIST_HIGH, (2,))
        if PRINTS: print(f"[INFO] Indexes: ({test_indexes[0]}, {test_indexes[1]})")
        start_img = train_set.data[test_indexes[0]]
        end_img = train_set.data[test_indexes[1]]
        base_imgs = [start_img, end_img]
        base_imgs = [img.type(torch.float32) for img in base_imgs]                  # data type
        base_imgs = [(img - torch.min(img)) / torch.max(img)  for img in base_imgs] # normalize
        base_imgs = [img.to(device=device) for img in base_imgs]                    # send to cpu
        base_imgs = [img.view(1, img.shape[0]*img.shape[1]).type(torch.FloatTensor) for img in base_imgs] # flatten

        # Encoding
        start_bottleneck = model.encode(base_imgs[0])
        end_bottleneck = model.encode(base_imgs[-1])

        interpolation_bottleneck = []
        for j in range(interpolation_steps):
            with torch.no_grad():
                num = j/(interpolation_steps-1)
                lerp = torch.lerp(start_bottleneck, end_bottleneck, j/(interpolation_steps-1))
            interpolation_bottleneck.append(lerp)

        # print("Start", start_bottleneck)
        # print("inter", interpolation_bottleneck)
        # print("end", end_bottleneck)
        # exit()

        # Decoding
        interpolation_imgs = [start_img]
        for img in interpolation_bottleneck:
            img = model.decode(img)
            img = img.view(28, 28).type(torch.FloatTensor)
            img = img.detach()
            img = img.cpu()
            img = img.numpy()
            interpolation_imgs.append(img)
        interpolation_imgs.append(end_img)

        imgs.append(interpolation_imgs)
    
    plot(imgs)
        
##### PLOTTING FUNCTION ######################################################################
def plot(imgs):
    """
    Plots a set of images
    """
    num_sets = len(imgs)
    img_per_set = len(imgs[0])
    if PRINTS: print(f"[INFO] num_sets: {num_sets}, img_per_set: {img_per_set}")

    f = plt.figure()
    for i in range(num_sets):
        for j in range(img_per_set):
            if PRINTS: print(f"[INFO] Img Num: {img_per_set*i+j+1}")
            f.add_subplot(num_sets, img_per_set, img_per_set*i+j+1)
            plt.imshow(imgs[i][j], cmap='gray')
    plt.show()

##### MAIN FUNCTION ##########################################################################
if __name__ == '__main__':
    #   read arguments from command line
    argParser = argparse.ArgumentParser()
    argParser.add_argument('-l', metavar='state', type=str, help='parameter file (.pth)')
    argParser.add_argument('-z', metavar='bottleneck size', type=int, help='int [32]')
    argParser.add_argument('-u', metavar='user_input', type=int, help='"0" for no input, ">0" for user interaction. int [32]')

    args = argParser.parse_args()

    save_file = None
    if args.l != None:
        save_file = args.l
    bottleneck_size = 8
    if args.z != None:
        bottleneck_size = args.z
    user_input = 0
    if args.u != None:
        user_input = args.u


    # Run Step 2
    visualize_MNIST_img(user_input)
    # Run Step 4
    autoencoder_test(user_input, save_file, bottleneck_size, noise=False)
    # Run Step 5
    autoencoder_test(user_input, save_file, bottleneck_size, noise=True)
    # Run Step 6
    bottleneck_interpolation(user_input, save_file)

    print("[INFO] Program finished.")
    

