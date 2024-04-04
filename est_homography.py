import numpy as np


def est_homography(X, Y):
    """
    Calculates the homography of two planes, from the plane defined by X
    to the plane defined by Y. In this assignment, X are the coordinates of the
    four corners of the soccer goal while Y are the four corners of the penn logo

    Input:
        X: 4x2 matrix of (x,y) coordinates of goal corners in video frame
        Y: 4x2 matrix of (x,y) coordinates of logo corners in penn logo
    Returns:
        H: 3x3 homogeneours transformation matrix s.t. Y ~ H*X

    """

    ##### STUDENT CODE START #####
    A = np.zeros([X.shape[0]*2, 9]) # 8 x 9 matrix
    

    for i in range(X.shape[0]):
        ## ax  is [-x, -y, -1, 0, 0, 0, xx', yx', x']
        ## ay is [0, 0, 0, -x, -y, -1, xy', yy', y']
        Hi = 2*i
        Hj = (2*i)+1
        A[Hi][0] = A[Hj][3] = -X[i][0] 
        A[Hi][1] = A[Hj][4] = -X[i][1]
        A[Hi][2] = A[Hj][5] = -1
        A[Hi][6] = X[i][0] * Y[i][0]
        A[Hi][7] = X[i][1] * Y[i][0]
        A[Hi][8] = Y[i][0]
        A[Hj][6] = X[i][0] * Y[i][1]
        A[Hj][7] = X[i][1] * Y[i][1]
        A[Hj][8] = Y[i][1]

    U, S, Vh = np.linalg.svd(A, full_matrices=True)

    print(A)
    print(S)
    print(Vh)
    print(Vh[-1,:])
    print('Shapes:',U.shape, S.shape, Vh.shape)
    #print(np.allclose(A, np.dot(U[:, :8] * S, Vh)))

    H = Vh[-1,:].reshape(9,1).reshape((3,3))
    print('H shape:', H.shape)
    ##### STUDENT CODE END #####

    return H
