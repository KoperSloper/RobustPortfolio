import numpy as np
import cvxpy as cp
from scipy.linalg import sqrtm

def robust_gmvp_ellipsoidal(Sigma_hat, epsilon, S_shape_matrix=None):
    """
    Solves the Robust GMVP under Ellipsoidal Uncertainty for the Covariance Matrix.
    (Feng and Palomar, 2016 - Lemma 14.4)
    
    Problem:
        Minimize    Tr(Sigma_hat @ (W + Z)) + epsilon * || S^(1/2) vec(W + Z) ||_2
        Subject to:
                    [ W    w ]
                    [ w.T  1 ]  >> 0  (Schur complement for W >= w*w.T)
                    
                    sum(w) == 1
                    w >= 0
                    Z >> 0
    
    Parameters:
    -----------
    Sigma_hat      : (N, N) array
                     The nominal covariance matrix estimate.
    epsilon        : float
                     Uncertainty radius.
    S_shape_matrix : (N^2, N^2) array, optional
                     The shape matrix S for the uncertainty ellipsoid defined 
                     on the VECTORIZED covariance.
                     
                     If None, defaults to Identity (Spherical Uncertainty in Frobenius norm),
                     which simplifies the penalty to epsilon * || W + Z ||_F.
    
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
    
    # Term B: Robust Penalty
    # The math requires vec(W) + vec(Z).
    # Standard math vec() stacks columns. We use order='F' (Fortran) to mimic this.
    vec_WZ = cp.reshape(W + Z, (N*N, 1), order='F')
    
    if S_shape_matrix is None:
        # Defaults to Spherical uncertainty: || vec(W+Z) ||_2  == || W+Z ||_F
        # This is much faster/more stable than building a huge Identity matrix.
        term_robust = epsilon * cp.norm(W + Z, "fro")
    else:
        # Full Ellipsoidal case: || S^1/2 @ vec(W+Z) ||_2
        # Note: S_shape_matrix must be (N^2, N^2). 
        # For N=50, N^2=2500, S is 2500x2500 (6.25M entries). Expensive!
        
        # We compute the sqrt of S outside CVXPY to keep the graph smaller
        try:
            S_sqrt = np.linalg.cholesky(S_shape_matrix)
        except:
            S_sqrt = sqrtm(S_shape_matrix)
            
        term_robust = epsilon * cp.norm(S_sqrt @ vec_WZ, 2)

    obj = cp.Minimize(term_nominal + term_robust)
    
    # --- 3. Constraints ---
    constraints = [
        # a. Budget and Long-only
        cp.sum(w) == 1,
        w >= 0,
        
        # b. Auxiliary PSD Matrix
        Z >> 0,
        
        # c. Relaxation Constraint (LMI)
        # This ensures W >= w @ w.T
        cp.bmat([
            [W,                     cp.reshape(w, (N, 1))],
            [cp.reshape(w, (1, N)), np.array([[1.0]])]
        ]) >> 0
    ]
    
    # --- 4. Solve ---
    prob = cp.Problem(obj, constraints)
    
    # This requires an SDP solver (SCS, CLARABEL, MOSEK)
    try:
        prob.solve(solver=cp.CLARABEL, verbose=False)
    except:
        prob.solve(solver=cp.SCS, verbose=False)
        
    if w.value is None:
        print("Optimization Failed.")
        return np.ones(N)/N
        
    # Cleanup
    w_final = w.value
    w_final[w_final < 1e-6] = 0
    return w_final / np.sum(w_final)