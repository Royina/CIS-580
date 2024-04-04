import numpy as np

def pose_candidates_from_E(E):
  transform_candidates = []
  ##Note: each candidate in the above list should be a dictionary with keys "T", "R"
  """ YOUR CODE HERE
  """
  U, S, Vt = np.linalg.svd(E, full_matrices = True)
  c = -1
  U_c = - U
  for i in range(2):
    U_c = c * U_c
    ## Rz(pi/2)
    T = U_c[:,-1]
    Rz = np.array([[0,-1,0],[1,0,0],[0,0,1]])
    R = U @ Rz.T @ Vt
    if np.linalg.det(R)<0:
      R = -R
      T = -T
    transform_candidates.append({'T':T,'R':R})

    

    ## Rz(-pi/2)
    T = U_c[:,-1]
    R = U @ Rz.T.T @ Vt
    if np.linalg.det(R)<0:
      R = -R
      T = -T
    transform_candidates.append({'T':T, 'R':R})
    

  """ END YOUR CODE
  """
  return transform_candidates