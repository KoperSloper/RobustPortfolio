import numpy as np
import cvxpy as cp
from scipy.linalg import sqrtm, cholesky

def robust_mvo_ellipsoidal(mu_hat, Sigma_hat, lam, epsilon, kappa, S_mu):
    """
    Solves the Robust MVO problem using SOCP/QP formulation (Orders of magnitude faster).
    
    Equivalent Objective:
        Minimize: w.T @ Sigma @ w + epsilon * (w.T @ inv(Sigma) @ w) - lambda * ret + penalty
    """
    N = len(mu_hat)
    
    # Pre-compute Inverse Covariance for the uncertainty term
    # Term: || P W P ||_fro  ==  w.T @ Sigma^-1 @ w
    try:
        Sigma_inv = np.linalg.inv(Sigma_hat)
    except:
        # Fallback for singular matrices
        Sigma_inv = np.linalg.pinv(Sigma_hat)

    # Variables (Vector only, no matrices!)
    w = cp.Variable(N)
    
    # 1. Standard Risk (w.T @ Sigma @ w)
    term_risk = cp.quad_form(w, Sigma_hat)
    
    # 2. Uncertainty Penalty (Approximated as w.T @ Sigma^-1 @ w)
    term_uncertainty = epsilon * cp.quad_form(w, Sigma_inv)
    
    # 3. Return Estimate
    term_ret_est = -lam * (w.T @ mu_hat)
    
    # 4. Mean Uncertainty Penalty (SOC term)
    # We use cp.norm(S_mu_sqrt @ w, 2) which is efficient
    try:
        S_mu_sqrt = np.array(cholesky(S_mu))
    except:
        S_mu_sqrt = np.array(sqrtm(S_mu))
        
    term_ret_penalty = lam * kappa * cp.norm(S_mu_sqrt.T @ w, 2)
    
    # Total Objective
    obj = cp.Minimize(term_risk + term_uncertainty + term_ret_est + term_ret_penalty)
    
    constraints = [
        cp.sum(w) == 1,
        w >= 0
    ]
    
    prob = cp.Problem(obj, constraints)
    
    # Use CLARABEL (Best open source) or OSQP (Fastest for QPs)
    # Since we have an SOC term (norm), CLARABEL or SCS is required. 
    # If you remove the norm term, OSQP would be instant.
    prob.solve(solver=cp.CLARABEL, verbose=False)
    
    if w.value is None:
        return np.ones(N)/N
        
    w_final = w.value
    w_final[w_final < 1e-5] = 0
    return w_final / np.sum(w_final)