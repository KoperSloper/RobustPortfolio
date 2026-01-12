import numpy as np
import cvxpy as cp
from scipy.linalg import sqrtm, cholesky

def robust_mvo_general_unified(mu_hat, Sigma_hat, lam, epsilon, kappa, S_mu, P_shape):
    """
    Solves the Robust MVO using the General SDP formulation with a custom Uncertainty Shape.
    
    Parameters:
    -----------
    P_shape : (N, N) symmetric matrix
        Defines the geometry of the uncertainty set for Covariance.
        The robust penalty term is: epsilon * || P_shape @ (W + Z) @ P_shape ||_F
        
        Possible choices for P_shape:
        1. "Regularization/Inverse" View: P = Sigma^-1/2 (Targets ill-conditioning)
        2. "Statistical" View:            P = Sigma^1/2  (Uncertainty scales with Volatility)
        3. "Spherical" View:              P = Identity   (Uncertainty is uniform)
    """
    N = len(mu_hat)
    
    # Mean Uncertainty Shape (S_mu^1/2)
    try:
        S_mu_sqrt = np.array(cholesky(S_mu))
    except:
        S_mu_sqrt = np.array(sqrtm(S_mu))

    # variables
    w = cp.Variable(N)
    W = cp.Variable((N, N), symmetric=True)
    Z = cp.Variable((N, N), symmetric=True) 

    # Objective Components
    # A. Nominal Risk: Trace(Sigma * (W + Z))
    term_trace = cp.trace(Sigma_hat @ (W + Z))
    
    # B. Robust Penalty: epsilon * || P(W + Z)P ||_F
    term_robust = epsilon * cp.norm(P_shape @ (W + Z) @ P_shape, "fro")
    
    # C. Return Components: -lambda * (ret - penalty)
    term_ret = -lam * (w.T @ mu_hat)
    term_penalty = lam * kappa * cp.norm(S_mu_sqrt.T @ w, 2)
    
    obj = cp.Minimize(term_trace + term_robust + term_ret + term_penalty)

    constraints = [
        cp.sum(w) == 1,
        w >= 0,
        Z >> 0,
        cp.bmat([[W, cp.reshape(w, (N, 1))],
                 [cp.reshape(w, (1, N)), np.array([[1.0]])]]) >> 0
    ]

    prob = cp.Problem(obj, constraints)
    prob.solve(solver=cp.CLARABEL, verbose=False)

    if w.value is None: return np.ones(N)/N
    return np.maximum(w.value, 0) / np.sum(np.maximum(w.value, 0))

def solve_robust_mvo_unified(mu_hat, Sigma_hat, lam, epsilon, kappa, Q_cov, S_mu):
    """
    The Fast Robust MVO Solver.
    
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