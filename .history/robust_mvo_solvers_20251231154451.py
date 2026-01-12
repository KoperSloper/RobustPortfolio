import numpy as np
import cvxpy as cp
from scipy.linalg import sqrtm

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
        S_mu_sqrt = np.linalg.cholesky(S_mu)
    except:
        S_mu_sqrt = sqrtm(S_mu)

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


def robust_mvo_fast_socp(mu_hat, Sigma_hat, lam, epsilon, kappa, S_mu):
    """
    Solves the Robust MVO using the Optimized SOCP formulation.
    Assumes uncertainty set is small enough that worst-case Sigma remains PSD.
    """
    N = len(mu_hat)
    
    # 1. Pre-compute Sigma Inverse
    try:
        Sigma_inv = np.linalg.inv(Sigma_hat)
    except:
        Sigma_inv = np.linalg.pinv(Sigma_hat)

    # 2. Mean Uncertainty Shape
    try:
        S_mu_sqrt = np.linalg.cholesky(S_mu)
    except:
        S_mu_sqrt = sqrtm(S_mu)

    # 3. Variables (Vector only)
    w = cp.Variable(N)
    
    # 4. Objective Components (Simplified)
    # Risk terms become quadratic forms
    term_risk = cp.quad_form(w, Sigma_hat)
    term_robust = epsilon * cp.quad_form(w, Sigma_inv)
    
    term_ret = -lam * (w.T @ mu_hat)
    term_penalty = lam * kappa * cp.norm(S_mu_sqrt.T @ w, 2)
    
    obj = cp.Minimize(term_risk + term_robust + term_ret + term_penalty)
    
    constraints = [
        cp.sum(w) == 1,
        w >= 0
    ]
    
    # Solves instantly with CLARABEL
    prob = cp.Problem(obj, constraints)
    prob.solve(solver=cp.CLARABEL, verbose=False)
    
    if w.value is None: return np.ones(N)/N
    return np.maximum(w.value, 0) / np.sum(np.maximum(w.value, 0))