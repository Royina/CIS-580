import numpy as np

def P3P(Pc, Pw, K=np.eye(3)):
    """
    Solve Perspective-3-Point problem, given correspondence and intrinsic

    Input:
        Pc: 4x2 numpy array of pixel coordinate of the April tag corners in (x,y) format
        Pw: 4x3 numpy array of world coordinate of the April tag corners in (x,y,z) format
    Returns:
        R: 3x3 numpy array describing camera orientation in the world (R_wc)
        t: (3,) numpy array describing camera translation in the world (t_wc)

    """

    ##### STUDENT CODE START #####

    # Invoke Procrustes function to find R, t
    # You may need to select the R and t that could transoform all 4 points correctly. 
    # R,t = Procrustes(Pc_3d, Pw[1:4])
    # c_Pc = np.concatenate([Pc, np.ones((Pc.shape[0], 1))], axis=1)
    # c_Pc[:,-1] = c_Pc[:,-1]*K[0,0]
    # c_Pc123 = np.matmul(np.linalg.inv(K), c_Pc[:3,:].T)
    c_Pc1 = np.array([(Pc[0,0] - K[0,2])/K[0,0], (Pc[0,1] - K[1,2])/K[0,0], 1])
    c_Pc2 = np.array([(Pc[1,0] - K[0,2])/K[0,0], (Pc[1,1] - K[1,2])/K[0,0], 1])
    c_Pc3 = np.array([(Pc[2,0] - K[0,2])/K[0,0], (Pc[2,1] - K[1,2])/K[0,0], 1])
    j1 = c_Pc1/np.linalg.norm(c_Pc1)
    j2 = c_Pc2/np.linalg.norm(c_Pc2)
    j3 = c_Pc3/np.linalg.norm(c_Pc3)
    alpha = np.arccos(np.dot(j2, j3))
    beta = np.arccos(np.dot(j1, j3))
    gamma = np.arccos(np.dot(j1, j2))

    cos_alpha = np.cos(alpha)
    cos_beta = np.cos(beta)
    cos_gamma = np.cos(gamma)

    a = np.linalg.norm(Pw[1,:] - Pw[2,:])
    b = np.linalg.norm(Pw[0,:] - Pw[2,:])
    c = np.linalg.norm(Pw[0,:] - Pw[1,:])


    A = np.zeros((5))
    a_minus_c = ((a**2 - c**2)/b**2)
    a_plus_c = ((a**2 + c**2)/b**2)
    A[4] = ((1+ a_minus_c)**2) - ((4*(a**2)/b**2)* (cos_gamma**2))
    A[3] = 4 * (-(a_minus_c * (1+a_minus_c)*cos_beta)+(2*((a**2)/(b**2))*cos_gamma*cos_gamma*cos_beta)-((1 - a_plus_c)*cos_alpha*cos_gamma))
    A[2] = 2 * ((a_minus_c**2)-1+(2*(a_minus_c**2)*(cos_beta**2))+(2*((b**2 - c**2)/b**2)*(cos_alpha**2))-(4*a_plus_c*cos_alpha*cos_beta*cos_gamma)+(2*((b**2 - a**2)/b**2)*(cos_gamma**2)))
    A[1] = 4 * ((a_minus_c*(1-a_minus_c)*cos_beta) - ((1-a_plus_c)*cos_alpha*cos_gamma) + (2*((c**2)/(b**2))*(cos_alpha**2)*cos_beta))
    A[0] = ((a_minus_c -1)**2) - (4*((c**2)/(b**2))*(cos_alpha**2))


    coefs = np.roots(A)
    coefs = coefs[np.where(np.isreal(coefs)==1)]
    coefs = np.real(coefs)
    
 
    X = np.zeros((3,3))
    min_val = 1e10
    min_R = np.zeros((3,3))
    min_t = np.zeros((3,))

    for i in range(len(coefs)): #len(coefs)
        v = coefs[i]
        u = ((-1+a_minus_c)*(v**2) - (2*(a_minus_c)*v*cos_beta) + 1 +a_minus_c)/(2*(cos_gamma - (v*cos_alpha)))
        s0 = np.sqrt((b**2)/(1+(v**2)-(2*v*cos_beta)))
        s1 = u * s0
        s2 = v * s0
        X[0] = (s0 * j1).reshape((1,3))
        X[1] = (s1 * j2).reshape((1,3))
        X[2] = (s2 * j3).reshape((1,3))
        R, t = Procrustes(X, Pw[:3,:])  

        p4 = np.matmul(K, np.matmul(R.T, Pw[3,:]) - np.matmul(R.T, t))
        p4 = p4/p4[2]
        c_Pc = np.concatenate([Pc, np.ones((Pc.shape[0], 1))], axis=1)
        dist = np.linalg.norm(c_Pc[3,:] - p4)
        if min_val>dist:
            min_val = dist
            min_R = R
            min_t = t

    R = min_R
    t = min_t

    # if ~flag:
    #     print('Didnt Find R and t :(')
    #     return np.zeros((3,3)), t

    ##### STUDENT CODE END #####

    return R, t

def Procrustes(X, Y):
    """
    Solve Procrustes: Y = RX + t

    Input:
        X: Nx3 numpy array of N points in camera coordinate (returned by your P3P)
        Y: Nx3 numpy array of N points in world coordinate
    Returns:
        R: 3x3 numpy array describing camera orientation in the world (R_wc)
        t: (3,) numpy array describing camera translation in the world (t_wc)

    """
    ##### STUDENT CODE START #####
    ## y is A and x is B
    Ycentroid = np.mean(Y, axis=0)
    Xcentroid = np.mean(X, axis=0)
    Ycentered = Y - Ycentroid.reshape(1,3)
    Xcentered = X - Xcentroid.reshape(1,3)
    U, S, Vh = np.linalg.svd(np.matmul(Ycentered.T, Xcentered), full_matrices=True)
    R = np.matmul(U, Vh)

    if np.linalg.det(R)<0:
         Vh[-1] *=-1
         R = np.matmul(U,Vh)

    t = np.mean(Y, axis=0) - np.matmul(R, np.mean(X, axis=0))

    ##### STUDENT CODE END #####

    return R, t

def main():
        a= np.array([[1, 2],[3, 4],[1, 5],[4, 4]])
        b= np.array([[1, 2, 1],[2, 5, 7],[1, 4, 3],[1, 1, 0]])
        P3P(a, b, np.eye(3))
  
  
if __name__=="__main__":
    main()

