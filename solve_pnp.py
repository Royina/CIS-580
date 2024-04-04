from est_homography import est_homography
import numpy as np

def PnP(Pc, Pw, K=np.eye(3)):
    """
    Solve Perspective-N-Point problem with collineation assumption, given correspondence and intrinsic

    Input:
        Pc: 4x2 numpy array of pixel coordinate of the April tag corners in (x,y) format
        Pw: 4x3 numpy array of world coordinate of the April tag corners in (x,y,z) format
    Returns:
        R: 3x3 numpy array describing camera orientation in the world (R_wc)
        t: (3, ) numpy array describing camera translation in the world (t_wc)

    """

    ##### STUDENT CODE START #####

    # Homography Approach
    # Following slides: Pose from Projective Transformation
    H = est_homography( Pw[:, :-1].reshape((4,2)), Pc)
 

    H_dash = np.matmul(np.linalg.inv(K), H)
    U, S, Vh = np.linalg.svd(H_dash[:,:-1], full_matrices=False)
    r_12 = np.matmul(U, Vh)
    r_1 = r_12[:,0].reshape((3))
    r_2 = r_12[:,1].reshape((3))
    lambda_1 = np.sum(S)/2



    r_3 = np.cross(r_1, r_2)
    R = np.hstack([r_1.reshape((3,1)), r_2.reshape((3,1)), r_3.reshape((3,1))])


    t = H_dash[:,-1]/lambda_1

    R= R.T
    t = - np.matmul(R, t)




    ##### STUDENT CODE END #####

    return R, t

# if __name__ == '__main__':
#     pw = np.random.randn(4,3)
#     pc = np.random.randn(4,2)
#     print(pw.shape, pc.shape)

#     PnP(pc,pw)

