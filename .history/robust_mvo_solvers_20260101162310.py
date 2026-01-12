import numpy as np
import cvxpy as cp
from scipy.linalg import sqrtm, cholesky

def robust_mvo_general_ellipsoidal(mu_hat, Sigma_hat, lam, epsilon, kappa, S_mu):
    """
    Solves the Robust MVO using the General SDP formulation (Matches derivation).
    Includes the dual variable Z to handle potential non-PSD worst-case scenarios.
    """
    N = len(mu_hat)
    
    # 1. Shape Matrix P (Sigma^-1/2)
    vals, vecs = np.linalg.eigh(Sigma_hat)
    vals = np.maximum(vals, 1e-6)
    P = vecs @ np.diag(1.0 / np.sqrt(vals)) @ vecs.T
    
    # 2. Mean Uncertainty Shape (S_mu^1/2)
    try:
        S_mu_sqrt = np.array(cholesky(S_mu))
    except:
        S_mu_sqrt = np.array(sqrtm(S_mu))

    # 3. Variables (Matrix-based)
    w = cp.Variable(N)
    W = cp.Variable((N, N), symmetric=True)
    Z = cp.Variable((N, N), symmetric=True) 

    # 4. Objective Components (Exact derivation)
    # Risk: Trace(Sigma * (W + Z)) + epsilon * || P(W + Z)P ||_F
    term_trace = cp.trace(Sigma_hat @ (W + Z))
    term_robust = epsilon * cp.norm(P @ (W + Z) @ P, "fro")
    
    # Return: -lambda * (ret - penalty)
    term_ret = -lam * (w.T @ mu_hat)
    term_penalty = lam * kappa * cp.norm(S_mu_sqrt.T @ w, 2)
    
    obj = cp.Minimize(term_trace + term_robust + term_ret + term_penalty)

    constraints = [
        cp.sum(w) == 1,
        w >= 0,
        Z >> 0,  # Z is a dual variable in the PSD cone
        # Relaxation: [W w; w.T 1] >> 0
        cp.bmat([[W, cp.reshape(w, (N, 1))],
                 [cp.reshape(w, (1, N)), np.array([[1.0]])]]) >> 0
    ]

    # Must use CLARABEL or SCS for SDPs
    prob = cp.Problem(obj, constraints)
    prob.solve(solver=cp.CLARABEL, verbose=False)

    if w.value is None: return np.ones(N)/N
    return np.maximum(w.value, 0) / np.sum(np.maximum(w.value, 0))

def solve_robust_mvo_unified(mu_hat, Sigma_hat, lam, epsilon, kappa, Q_cov, S_mu):
    """
    The Universal Fast Robust MVO Solver.
    
    Parameters:
    -----------
    Q_cov : (N, N) matrix
        The shape of the covariance uncertainty. 
        Assumes the full uncertainty matrix is S_Sigma = Q_cov (x) Q_cov.
        - Use Sigma_inv for "Condition Number/Relative" robustness (Recommended).
        - Use Sigma for "Statistical" robustness.
        - Use Identity for "Spherical" robustness.
        
    S_mu : (N, N) matrix
        The covariance of the mean estimates (usually Sigma / T).
    """
    N = len(mu_hat)
    
    # 1. Prepare Mean Uncertainty (Square Root of S_mu)
    # Term: || S_mu^1/2 * w ||
    try:
        S_mu_sqrt = np.array(cholesky(S_mu))
    except:
        S_mu_sqrt = np.array(sqrtm(S_mu))

    # 2. Variables (Vector only - Fast!)
    w = cp.Variable(N)
    
    # --- OBJECTIVE COMPONENTS ---
    
    # A. Nominal Risk: w.T @ Sigma @ w
    term_risk = cp.quad_form(w, Sigma_hat)
    
    # B. Unified Robust Covariance Penalty
    # We simplified the complex matrix norm || (QxQ)^1/2 vec(ww.T) || 
    # down to a simple quadratic form: w.T @ Q @ w
    if np.array_equal(Q_cov, np.eye(N)):
        # Special case for Identity to be slightly faster
        term_robust = epsilon * cp.sum_squares(w)
    else:
        term_robust = epsilon * cp.quad_form(w, Q_cov)
    
    # C. Return Estimate
    term_ret = -lam * (w.T @ mu_hat)
    
    # D. Mean Uncertainty Penalty (SOC Term)
    term_mean_penalty = lam * kappa * cp.norm(S_mu_sqrt.T @ w, 2)
    
    # Combined Objective
    obj = cp.Minimize(term_risk + term_robust + term_ret + term_mean_penalty)
    
    constraints = [
        cp.sum(w) == 1,
        w >= 0
    ]
    
    # Solve with CLARABEL (Handles SOCP + QP efficiently)
    prob = cp.Problem(obj, constraints)
    prob.solve(solver=cp.CLARABEL, verbose=False)
    
    if w.value is None:
        return np.ones(N)/N
    
    # Cleanup numerical noise
    w_final = w.value
    w_final[w_final < 1e-6] = 0
    return w_final / np.sum(w_final)