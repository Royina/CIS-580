from lse import least_squares_estimation
import numpy as np

def ransac_estimator(X1, X2, num_iterations=60000):
    sample_size = 8

    eps = 10**-4

    best_num_inliers = -1
    best_inliers = None
    best_E = None
    e3 = np.array([[0,-1,0],[1,0,0],[0,0,0]])

    for i in range(num_iterations):
        # permuted_indices = np.random.permutation(np.arange(X1.shape[0]))
        permuted_indices = np.random.RandomState(seed=(i*10)).permutation(np.arange(X1.shape[0]))
        sample_indices = permuted_indices[:sample_size]
        test_indices = permuted_indices[sample_size:]

        """ YOUR CODE HERE
        """
        E = least_squares_estimation(X1[sample_indices], X2[sample_indices])
        # print(np.linalg.norm(np.matmul(e3, np.matmul(E, X1[test_indices,:].T)), axis=0)[:3]**2)
        # print(np.linalg.norm(e3 @ E @ X1[test_indices[0]])**2)
        # break

        ## vectorized implementation
        d_x2_x1 = np.diag(np.matmul(X2[test_indices,:], np.matmul(E, X1[test_indices,:].T)))**2/np.linalg.norm(np.matmul(e3, np.matmul(E, X1[test_indices,:].T)), axis=0)**2
        d_x1_x2 = np.diag(np.matmul(X1[test_indices,:], np.matmul(E.T, X2[test_indices,:].T)))**2/np.linalg.norm(np.matmul(e3, np.matmul(E.T, X2[test_indices,:].T)), axis=0)**2
        d = d_x2_x1.reshape((len(test_indices))) + d_x1_x2.reshape((len(test_indices)))
        idx = np.argwhere(d<eps)
        inliers = test_indices[idx[:,0]]
        inliers = np.append(sample_indices, inliers)

        ## loop implementation
        # inliers = []
        # for j, idx in enumerate(test_indices):
        #     d_x2_x1 = np.linalg.norm(X2[idx].T @ E @ X1[idx])**2/ np.linalg.norm(e3 @ E @ X1[idx])**2
        #     d_x1_x2 = np.linalg.norm(X1[idx].T @ E.T @ X2[idx])**2/ np.linalg.norm(e3 @ E.T @ X2[idx])**2
        #     d = d_x2_x1 + d_x1_x2
        #     if d < eps:
        #         inliers.append(idx)

        # inliers = np.append(sample_indices, np.array(inliers))
        """ END YOUR CODE
        """
        if inliers.shape[0] > best_num_inliers:
            best_num_inliers = inliers.shape[0]
            best_E = E
            best_inliers = inliers

    return best_E, best_inliers