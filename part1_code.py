import numpy as np
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import imageio.v2 as imageio
import time
import gdown

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

url = "https://drive.google.com/file/d/1rD1aaxN8aSynZ8OPA7EI3G936IF0vcUt/view?usp=sharing"
gdown.download(url=url, output='starry_night.jpg', quiet=False, fuzzy=True)

# Load painting image
painting = imageio.imread("starry_night.jpg")
painting = torch.from_numpy(np.array(painting, dtype=np.float32)/255.).to(device)
height_painting, width_painting = painting.shape[:2]

def positional_encoding(x, num_frequencies=6, incl_input=True):

    """
    Apply positional encoding to the input.

    Args:
    x (torch.Tensor): Input tensor to be positionally encoded.
      The dimension of x is [N, D], where N is the number of input coordinates,
      and D is the dimension of the input coordinate.
    num_frequencies (optional, int): The number of frequencies used in
     the positional encoding (default: 6).
    incl_input (optional, bool): If True, concatenate the input with the
        computed positional encoding (default: True).

    Returns:
    (torch.Tensor): Positional encoding of the input tensor.
    """

    results = []
    # if incl_input:
    #     results.append(x)
    #############################  TODO 1(a) BEGIN  ############################
    # encode input tensor and append the encoded tensor to the list of results.
    # freq = torch.pow(2,torch.arange(num_frequencies)).to(device)
    # sin_list = torch.sin(torch.pi*freq*x.reshape((-1,1))).to(device)
    # cos_list  = torch.cos(torch.pi*freq*x.reshape((-1,1))).to(device)
    # results.append(sin_list.reshape((x.shape[0],-1)))
    # results.append(cos_list.reshape((x.shape[0],-1)))
    # ############################  TODO 1(a) END  ##############################
    # return torch.cat(results, dim=-1)

    prepend = 1 if incl_input else 0
    enc_sz = x.shape[1] * (prepend + 2 * num_frequencies)
    res = torch.zeros((x.shape[0], enc_sz), device=x.device)
    if incl_input:
        res[:, :x.shape[1]] = x
    powers = torch.pow(2, torch.arange(num_frequencies, device=x.device)) # (L,)
    sin_phases = powers[None, :, None] * torch.pi * x[:, None, :] # (N, L, D)
    cos_phases = torch.pi / 2 - sin_phases
    phases = torch.stack([sin_phases, cos_phases], dim=-2) # (N, L, 2, D)
    flat = phases.flatten(1)
    res[:, prepend*x.shape[1]:] = torch.sin(flat)
    return res

class model_2d(nn.Module):

    """
    Define a 2D model comprising of three fully connected layers,
    two relu activations and one sigmoid activation.
    """

    def __init__(self, filter_size=128, num_frequencies=6):
        super().__init__()
        #############################  TODO 1(b) BEGIN  ############################
        # for autograder compliance, please follow the given naming for your layers
        self.layer_in = nn.Linear(4*num_frequencies+2, filter_size)
        self.layer = nn.Linear(filter_size, filter_size)
        self.layer_out = nn.Linear(filter_size,3)

        #############################  TODO 1(b) END  ##############################

    def forward(self, x):
        #############################  TODO 1(b) BEGIN  ############################
        # example of forward through a layer: y = self.layer_in(x)
        out = self.layer_in(x)
        out = F.relu(out)
        out = self.layer(out)
        out = F.relu(out)
        out = self.layer_out(out)
        x = F.sigmoid(out)
        #############################  TODO 1(b) END  ##############################
        return x

def normalize_coord(height, width, num_frequencies=6):

    """
    Creates the 2D normalized coordinates, and applies positional encoding to them

    Args:
    height (int): Height of the image
    width (int): Width of the image
    num_frequencies (optional, int): The number of frequencies used in
      the positional encoding (default: 6).

    Returns:
    (torch.Tensor): Returns the 2D normalized coordinates after applying positional encoding to them.
    """

    #############################  TODO 1(c) BEGIN  ############################
    # Create the 2D normalized coordinates, and apply positional encoding to them
    coords = torch.stack(torch.meshgrid(torch.arange(height)/height, torch.arange(width)/width), -1).reshape((-1,2))
    embedded_coordinates = positional_encoding(coords, num_frequencies)
    #############################  TODO 1(c) END  ############################

    return embedded_coordinates

def train_2d_model(test_img, num_frequencies, device, model=model_2d, positional_encoding=positional_encoding, show=True):

    # Optimizer parameters
    lr = 5e-4
    iterations = 10000
    height, width = test_img.shape[:2]

    # Number of iters after which stats are displayed
    display = 2000

    # Define the model and initialize its weights.
    model2d = model(num_frequencies=num_frequencies)


    def weights_init(m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)

    model2d.apply(weights_init)
    model2d.to(device)

    #############################  TODO 1(c) BEGIN  ############################
    # Define the optimizer
    optimizer = torch.optim.Adam(model2d.parameters(), lr=lr)
    criterion = nn.MSELoss()
    #############################  TODO 1(c) END  ############################

    # Seed RNG, for repeatability
    seed = 5670
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Lists to log metrics etc.
    psnrs = []
    iternums = []

    t = time.time()
    t0 = time.time()

    #############################  TODO 1(c) BEGIN  ############################
    # Create the 2D normalized coordinates, and apply positional encoding to them
    embed_coords = normalize_coord(height, width, num_frequencies)
    #############################  TODO 1(c) END  ############################

    for i in range(iterations+1):
        optimizer.zero_grad()
        #############################  TODO 1(c) BEGIN  ############################
        # Run one iteration
        pred = model2d(embed_coords.to(device=device))
        pred = pred.reshape(height, width, 3)
        loss = criterion(pred, test_img)
        loss.backward()
        optimizer.step()
        # Compute mean-squared error between the predicted and target images. Backprop!



        #############################  TODO 1(c) END  ############################

        # Display images/plots/stats
        if i % display == 0 and show:
            #############################  TODO 1(c) BEGIN  ############################
            # Calculate psnr
            psnr = 10 * torch.log10((torch.max(test_img)**2)/loss.item())
            #############################  TODO 1(c) END  ############################

            print("Iteration %d " % i, "Loss: %.4f " % loss.item(), "PSNR: %.2f" % psnr.item(), \
                "Time: %.2f secs per iter" % ((time.time() - t) / display), "%.2f secs in total" % (time.time() - t0))
            t = time.time()

            psnrs.append(psnr.item())
            iternums.append(i)

            plt.figure(figsize=(13, 4))
            plt.subplot(131)
            plt.imshow(pred.detach().cpu().numpy())
            plt.title(f"Iteration {i}")
            plt.subplot(132)
            plt.imshow(test_img.cpu().numpy())
            plt.title("Target image")
            plt.subplot(133)
            plt.plot(iternums, psnrs)
            plt.title("PSNR")
            plt.show()

        # if i%2500==0 and i>0 and i<7000:
        #   lr = lr * 0.25
        #   for op_params in optimizer.param_groups:
        #     op_params['lr'] = lr


    print('Done!')
    torch.save(model2d.state_dict(),'model_2d_' + str(num_frequencies) + 'freq.pt')
    plt.imsave('van_gogh_' + str(num_frequencies) + 'freq.png',pred.detach().cpu().numpy())
    return pred.detach().cpu()

