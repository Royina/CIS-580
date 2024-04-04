import numpy as np
import matplotlib.pyplot as plt

def plot_flow(image, flow_image, confidence, threshmin=10):
    """
    Plot a flow field of one frame of the data.
    
    Inputs:
        - image: grayscale image - shape: (H, W)
        - flow_image: optical flow - shape: (H, W, 2)
        - confidence: confidence of the flow estimates - shape: (H, W)
        - threshmin: threshold for confidence (optional) - scalar
    """
    
    ### STUDENT CODE START ###
    
    # Useful function: np.meshgrid()
    # Hint: Use plt.imshow(<your image>, cmap='gray') to display the image in grayscale
    # Hint: Use plt.quiver(..., color='red') to plot the flow field on top of the image in a visible manner
    plt.imshow(image, cmap = 'gray')
    # flow_images_array = np.zeros_like(flow_image)
    # for i in range(flow_image.shape[0]):
    #     for j in range(flow_image.shape[1]):
    #         if confidence[i,j] > threshmin:
    #             flow_images_array[i, j] = flow_image[i,j]
    # x, y = np.meshgrid(np.arange(0, image.shape[0]), np.arange(0, image.shape[1]))
    y,x = np.where(confidence>threshmin)
    plt.quiver(x, y, flow_image[y,x,0], -flow_image[y,x,1], color='red', scale=25)
    # plt.quiver(x, y, flow_images_array[:,:,0], -flow_images_array[:,:,1], color='red')
    # plt.show()

    
    ### STUDENT CODE END ###

    # this function has no return value
    return





    

