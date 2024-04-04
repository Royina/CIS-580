import numpy as np

def flow_lk_patch(Ix, Iy, It, x, y, size=5):
    """
    Find the Lucas-Kanade optical flow on a single square patch.
    The patch is centered at (y, x), therefore it generally extends
    from x-size//2 to x+size//2 (inclusive), same for y, EXCEPT when
    exceeding image boundaries!
    
    WARNING: Pay attention to how you index the images! The first coordinate
    is actually the y-coordinate, and the second coordinate is the x-coordinate.
    
    Inputs:
        - Ix: image gradient along the x-dimension - shape: (H, W)
        - Iy: image gradient along the y-dimension - shape: (H, W)
        - It: image time-derivative - shape: (H, W)
        - x: SECOND coordinate of patch center - integer in range [0, W-1]
        - y: FIRST coordinate of patch center - integer in range [0, H-1]
        - size: optional parameter to change the side length of the patch in pixels
    
    Outputs:
        - flow: flow estimate for this patch - shape: (2,)
        - conf: confidence of the flow estimates - scalar
    """

    ### STUDENT CODE START ###
    Ix = Ix[y-size//2:y+size//2+1, x-size//2:x+size//2+1].flatten()
    Iy =  Iy[y-size//2:y+size//2+1, x-size//2:x+size//2+1].flatten()
    It =  It[y-size//2:y+size//2+1, x-size//2:x+size//2+1].flatten()
    A = np.vstack([Ix, Iy]).T
    B = -It.reshape((-1,1))
    flow, _, _, conf = np.linalg.lstsq(A, B, rcond=-1)
    flow = flow.reshape((2,))
    conf = conf.min()

   
    ### STUDENT CODE END ###

    return flow, conf


def flow_lk(Ix, Iy, It, size=5):
    """
    Compute the Lucas-Kanade flow for all patches of an image.
    To do this, iteratively call flow_lk_patch for all possible patches.
    
    WARNING: Pay attention to how you index the images! The first coordinate
    is actually the y-coordinate, and the second coordinate is the x-coordinate.
    
    Inputs:
        - Ix: image gradient along the x-dimension - shape: (H, W)
        - Iy: image gradient along the y-dimension - shape: (H, W)
        - It: image time-derivative
    Outputs:
        - image_flow: flow estimate for each patch - shape: (H, W, 2)
        - confidence: confidence of the flow estimates - shape: (H, W)
    """

    ### STUDENT CODE START ###
    
    image_flow = np.zeros((Ix.shape[0], Ix.shape[1], 2))
    confidence = np.zeros((Ix.shape[0], Ix.shape[1]))

    Ix_pad = np.pad(Ix, (size//2, size//2), 'constant', constant_values=(0,0))
    Iy_pad =  np.pad(Iy, (size//2, size//2), 'constant', constant_values=(0,0))
    It_pad =  np.pad(It, (size//2, size//2), 'constant', constant_values=(0,0))
    # double for-loop to iterate over all patches
    for y in range(Ix.shape[0]):
        y_new = y + size//2
        for x in range(Ix.shape[1]):
            x_new = x + size//2
            flow, conf = flow_lk_patch(Ix_pad, Iy_pad, It_pad, x_new, y_new, size)
            # flow, conf = flow_lk_patch(Ix, Iy, It, x, y, size)
            image_flow[y, x,:] = flow
            confidence[y, x] = conf
    ### STUDENT CODE END ###
    
    print(image_flow.shape)
    print(confidence.shape)
    return image_flow, confidence



    

