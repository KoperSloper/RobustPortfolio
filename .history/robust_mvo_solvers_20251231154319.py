import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt
from scipy.linalg import sqrtm
from scipy.stats import chi2
import time

# -----------------------------------------------------------------------------
# OPTION A: General Derivation (Slow, Theoretically Complete)
# -----------------------------------------------------------------------------
def solve_robust_mvo_general_sdp(mu_hat, Sigma_hat, lam, epsilon, kappa, S_mu):
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


# -----------------------------------------------------------------------------
# OPTION B: Fast Implementation (Optimized for Finance)
# -----------------------------------------------------------------------------
def solve_robust_mvo_fast_socp(mu_hat, Sigma_hat, lam, epsilon, kappa, S_mu):
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


# -----------------------------------------------------------------------------
# Comparison
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    np.random.seed(42)
    N = 40
    # Create dummy financial data
    mu = np.random.uniform(0.05, 0.15, N)
    A = np.random.randn(N, N)
    Sigma = A @ A.T + np.eye(N)*0.01
    
    # Robust Params
    lam = 5.0
    eps = 0.5
    kappa = 1.0
    S_mu = Sigma / 252.0

    print(f"Comparing performance for N={N} assets...\n")

    # Run Option A
    t0 = time.time()
    w_gen = solve_robust_mvo_general_sdp(mu, Sigma, lam, eps, kappa, S_mu)
    t_gen = time.time() - t0
    print(f"Option A (General SDP): {t_gen:.4f} seconds")

    # Run Option B
    t0 = time.time()
    w_fast = solve_robust_mvo_fast_socp(mu, Sigma, lam, eps, kappa, S_mu)
    t_fast = time.time() - t0
    print(f"Option B (Fast SOCP):   {t_fast:.4f} seconds")

    # Verify Results Match
    diff = np.linalg.norm(w_gen - w_fast)
    print(f"\nEuclidean Distance between weight vectors: {diff:.6f}")
    if diff < 1e-3:
        print(">> SUCCESS: Solutions are identical.")
    else:
        print(">> NOTE: Solutions differ (Check epsilon magnitude).")