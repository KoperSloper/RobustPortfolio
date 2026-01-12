import numpy as np
import cvxpy as cp
from scipy.linalg import sqrtm, cholesky

def robust_mvo(mu_hat, Sigma_hat, lam, epsilon, kappa, S_mu):
    """
    Solves the Robust MVO problem using the Explicit SDP formulation.
    
    Variables:
        w (N,): Portfolio weights.
        W (N,N): Symmetric matrix approximating w*w.T.
        Z (N,N): Symmetric auxiliary matrix for uncertainty.
        
    Objective:
        Minimize: Trace(Sigma*(W+Z)) + epsilon*||P(W+Z)P|| - lambda*Return + lambda*kappa*||S_mu^1/2 w||
    """
    N = len(mu_hat)
    
    # A. Covariance Uncertainty Shape (P = Sigma^-1/2)
    vals, vecs = np.linalg.eigh(Sigma_hat)
    vals = np.maximum(vals, 1e-6)
    inv_sqrt_vals = 1.0 / np.sqrt(vals)
    P = vecs @ np.diag(inv_sqrt_vals) @ vecs.T
    
    # B. Mean Uncertainty Shape (S_mu^1/2)
    try:
        S_mu_sqrt = np.array(cholesky(S_mu)
    except:
        S_mu_sqrt = np.array(sqrtm(S_mu))

    # variables
    w = cp.Variable(N)
    W = cp.Variable((N, N), symmetric=True)
    Z = cp.Variable((N, N), symmetric=True)
    
    # objective components
    # Tr( Sigma * (W + Z) )
    term_trace = cp.trace(Sigma_hat @ (W + Z))
    
    # epsilon * || P(W + Z)P ||_fro
    term_norm_cov = epsilon * cp.norm(P @ (W + Z) @ P, "fro")
    
    term_ret_est = -lam * (w.T @ mu_hat)

    # lambda * kappa * || S_mu^1/2 * w ||_2
    term_ret_penalty = lam * kappa * cp.norm(S_mu_sqrt.T @ w, 2)
    
    obj = cp.Minimize(term_trace + term_norm_cov + term_ret_est + term_ret_penalty)

    constraints = [
        # 1. Standard Portfolio Constraints
        cp.sum(w) == 1,
        w >= 0,
        
        # 2. Schur Complement for W (Relaxation of W = ww^T)
        # [ W   w ]
        # [ w.T 1 ]  >> 0
        cp.bmat([
        [W, cp.reshape(w, (N, 1), order='C')],
        [cp.reshape(w, (1, N), order='C'), np.array([[1.0]])]
    ]) >> 0,
        
        # 3. Non-negativity for Z
        Z >> 0
    ]

    prob = cp.Problem(obj, constraints)
    
    prob.solve(solver=cp.SCS, verbose=False, eps=1e-4, acceleration_lookback=0)

    if w.value is None:
        return np.ones(N)/N
    
    # Clean weights
    w_final = w.value
    w_final[w_final < 1e-5] = 0
    return w_final / np.sum(w_final)