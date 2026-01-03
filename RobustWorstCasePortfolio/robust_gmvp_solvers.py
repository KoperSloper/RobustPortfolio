import numpy as np
import cvxpy as cp
from scipy.linalg import sqrtm

def robust_gmvp_ellipsoidal(Sigma_hat, epsilon, S_shape_matrix=None):
    """
    Solves the Robust Global Minimum Variance Portfolio (GMVP) under Ellipsoidal Uncertainty.

    Problem Formulation:
    --------------------
    Minimize:   Tr(Sigma_hat @ (W + Z)) + epsilon * || S^(1/2) @ vec(W + Z) ||_2
    Subject to: [ W    w ]
                [ w.T  1 ] >> 0  (Schur complement for W >= w*w.T)
                sum(w) == 1
                w >= 0
                Z >> 0

    Parameters:
    -----------
    Sigma_hat : (N, N) array
        The nominal covariance matrix estimate.
    epsilon : float
        Uncertainty radius.
    S_shape_matrix : (N^2, N^2) array, optional
        The shape matrix S for the uncertainty ellipsoid defined on the VECTORIZED covariance.
        If None, defaults to Identity (Spherical Uncertainty), simplifying the penalty 
        to epsilon * || W + Z ||_F.

    Returns:
    --------
    w : (N,) array
        Optimal portfolio weights.
    """
    N = len(Sigma_hat)

    w = cp.Variable(N)
    W = cp.Variable((N, N), symmetric=True)
    Z = cp.Variable((N, N), symmetric=True)
    
    # Trace(Sigma * (W + Z))
    term_nominal = cp.trace(Sigma_hat @ (W + Z))
    
    vec_WZ = cp.reshape(W + Z, (N*N, 1), order='F')
    
    if S_shape_matrix is None:
        term_robust = epsilon * cp.norm(W + Z, "fro")
    else:
        # Full Ellipsoidal case: || S^1/2 @ vec(W+Z) ||_2
        try:
            S_sqrt = np.linalg.cholesky(S_shape_matrix)
        except:
            S_sqrt = sqrtm(S_shape_matrix)
            
        term_robust = epsilon * cp.norm(S_sqrt @ vec_WZ, 2)

    obj = cp.Minimize(term_nominal + term_robust)

    constraints = [
        cp.sum(w) == 1,
        w >= 0,
        
        Z >> 0,
        
        cp.bmat([
            [W,                     cp.reshape(w, (N, 1))],
            [cp.reshape(w, (1, N)), np.array([[1.0]])]
        ]) >> 0
    ]

    prob = cp.Problem(obj, constraints)
    
    try:
        prob.solve(solver=cp.CLARABEL, verbose=False)
    except:
        prob.solve(solver=cp.SCS, verbose=False)
        
    if w.value is None:
        print("Optimization Failed.")
        return np.ones(N)/N
        
    w_final = w.value
    w_final[w_final < 1e-6] = 0
    return w_final / np.sum(w_final)