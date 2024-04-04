import numpy as np

def depth(flow, confidence, ep, K, thres=10):
    """
    Compute the depth map from the flow and confidence map.
    
    Inputs:
        - flow: optical flow - shape: (H, W, 2)
        - confidence: confidence of the flow estimates - shape: (H, W)
        - ep: epipole - shape: (3,)
        - K: intrinsic calibration matrix - shape: (3, 3)
        - thres: threshold for confidence (optional) - scalar
    
    Output:
        - depth_map: depth at every pixel - shape: (H, W)
    """
    depth_map = np.zeros_like(confidence)

    ### STUDENT CODE START ###
    
    # 1. Find where flow is valid (confidence > threshold)
    # 2. Convert these pixel locations to normalized projective coordinates
    # 3. Same for epipole and flow vectors
    # 4. Now find the depths using the formula from the lecture slides
    
    
    ### STUDENT CODE END ###
    good_idx = np.flatnonzero(confidence>thres)
    permuted_indices = np.random.RandomState(seed=10).permutation(
        good_idx
    )
    valid_idx=permuted_indices[:3000]

    x_valid = good_idx//confidence.shape[1]
    y_valid = good_idx%confidence.shape[1]
    x, y = np.meshgrid(np.arange(0,flow.shape[1]),np.arange(0,flow.shape[0]))

    K_inv = np.linalg.inv(K)
    eps_norm = np.matmul(K_inv, ep.reshape((3,1)))
    # eps_norm = eps_norm/eps_norm[-1,:]

    flow_x = flow[:,:,0].flatten()
    flow_y = flow[:,:,1].flatten()
    u = np.vstack([flow_x, flow_y, np.zeros_like(flow_y)]).T
    u_norm = np.matmul(K_inv, u.T)

    # x_new = x - 256
    # y_new = y - 256
    xp = np.vstack([x.flatten(), y.flatten(), np.ones((x.flatten().shape[0]))]).T
    xp_norm = np.matmul(K_inv, xp.T)

    xp_cross_u_norm = np.cross(xp_norm[:,valid_idx].T, u_norm[:,valid_idx].T)
    U, S, Vh = np.linalg.svd(xp_cross_u_norm)
    Vz =np.abs( Vh[-1,:][-1])

    numerator = np.linalg.norm(xp_norm - eps_norm, axis=0)
    denominator = np.linalg.norm(u_norm, axis=0)

    depth = (numerator/denominator) * Vz
    depth_valid = depth[good_idx]
    # depth_map = depth.reshape((flow.shape[0], flow.shape[1]))

    for i in range(len(x_valid)):
        j = x_valid[i]
        k = y_valid[i]
        depth_map[j,k] = depth_valid[i]
        
    
    ## Truncate the depth map to remove outliers
    
    # require depths to be positive
    truncated_depth_map = np.maximum(depth_map, 0) 
    valid_depths = truncated_depth_map[truncated_depth_map > 0]
    
    # You can change the depth bound for better visualization if you depth is in different scale
    depth_bound = valid_depths.mean() + 10 * np.std(valid_depths)
    print(f'depth bound: {depth_bound}')

    # set depths above the bound to 0 and normalize to [0, 1]
    truncated_depth_map[truncated_depth_map > depth_bound] = 0
    truncated_depth_map = truncated_depth_map / truncated_depth_map.max()

    return truncated_depth_map
