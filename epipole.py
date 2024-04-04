import numpy as np

def epipole(flow_x, flow_y, smin, thresh, num_iterations=None):
    """
    Compute the epipole from the flows,
    
    Inputs:
        - flow_x: optical flow on the x-direction - shape: (H, W)
        - flow_y: optical flow on the y-direction - shape: (H, W)
        - smin: confidence of the flow estimates - shape: (H, W)
        - thresh: threshold for confidence - scalar
    	- Ignore num_iterations
    Outputs:
        - ep: epipole - shape: (3,)
    """
    # Logic to compute the points you should use for your estimation
    # We only look at image points above the threshold in our image
    # Due to memory constraints, we cannot use all points on the autograder
    # Hence, we give you valid_idx which are the flattened indices of points
    # to use in the estimation estimation problem 
    good_idx = np.flatnonzero(smin>thresh)
    permuted_indices = np.random.RandomState(seed=10).permutation(
        good_idx
    )
    valid_idx=permuted_indices[:3000]

    ### STUDENT CODE START - PART 1 ###
    
    # 1. For every pair of valid points, compute the epipolar line (use np.cross)
    # Hint: for faster computation and more readable code, avoid for loops! Use vectorized code instead.
    flow_x_selected  = flow_x.flatten()[valid_idx]
    flow_y_selected = flow_y.flatten()[valid_idx]
    u = np.vstack([flow_x_selected, flow_y_selected, np.zeros_like(flow_y_selected)]).T

    # index_mat = np.zeros_like(smin).flatten()
    # index_mat[valid_idx] = 1
    # index_mat = index_mat.reshape(smin.shape)
    # xp = np.argwhere(index_mat == 1)
    # xp[:, [1, 0]] = xp[:, [0, 1]] ## interchanging y and x to make it (x,y)
    x = valid_idx//smin.shape[1]
    y = valid_idx%smin.shape[1]
    x = x - 256
    y = y-256
    xp = np.vstack([y,x, np.ones((x.shape[0]))]).T

    xp_cross_u = np.cross(xp, u)
    U, S, Vh = np.linalg.svd(xp_cross_u)
    ep = Vh[-1,:]
    

    
 
    return ep