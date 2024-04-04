import numpy as np

def least_squares_estimation(X1, X2):
  """ YOUR CODE HERE
  """
  p = np.expand_dims(X1.T, axis=1)
  q = np.expand_dims(X2.T, axis=0)
  A = np.repeat(q,3,axis=0) * np.repeat(p,3,axis=1)
  A = A.T.reshape(X1.shape[0],9)
  U, S, Vt = np.linalg.svd(A, full_matrices=True)
  E_estimate = Vt[-1,:]
  E_estimate = E_estimate.reshape((3,3))
  U, S, Vt = np.linalg.svd(E_estimate, full_matrices=True)
  S_new = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 0]])
  E = np.matmul(U, np.matmul(S_new, Vt))
  """ END YOUR CODE
  """
  
  return E
