import numpy as np
import cvxpy as cp
from scipy.linalg import sqrtm, cholesky

def robust_mvo_general_ellipsoid(mu_hat, Sigma_hat, lam, epsilon, kappa, S_mu, P_shape):
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
        cp.bmat([
            [W, cp.reshape(w, (N, 1), order='C')], 
            [cp.reshape(w, (1, N), order='C'), np.array([[1.0]])] 
        ]) >> 0
    ]

    prob = cp.Problem(obj, constraints)
    prob.solve(solver=cp.CLARABEL, verbose=False)

    if w.value is None: return np.ones(N)/N
    return np.maximum(w.value, 0) / np.sum(np.maximum(w.value, 0))

# Since for any reasonable uncertainty region, Z will be driven to zero, we can simplify the problem:
def robust_mvo_fast_ellipsoid(mu_hat, Sigma_hat, lam, epsilon, kappa, Q_cov, S_mu):
    """
    The Fast(er) Robust MVO Solver.
    
    Parameters:
    -----------
    Q_cov : (N, N) matrix
        The shape of the covariance uncertainty. 
        Assumes the full uncertainty matrix is S_Sigma = Q_cov (x) Q_cov.
        - Use Sigma_inv for "Condition Number/Relative" robustness.
        - Use Sigma for "Statistical" robustness.
        - Use Identity for "Spherical" robustness.
        
    S_mu : (N, N) matrix
        The covariance of the mean estimates (usually Sigma / T).
    """
    N = len(mu_hat)
    
    try:
        S_mu_sqrt = np.array(cholesky(S_mu))
    except:
        S_mu_sqrt = np.array(sqrtm(S_mu))

    # variables (Vector only)
    w = cp.Variable(N)
        
    # A. Nominal Risk: w.T @ Sigma @ w
    term_risk = cp.quad_form(w, Sigma_hat)
    
    # B. Unified Robust Covariance Penalty
    # We simplified the complex matrix norm || (QxQ)^1/2 vec(ww.T) || 
    # down to a simple quadratic form: w.T @ Q @ w
    if np.array_equal(Q_cov, np.eye(N)):
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
    
    prob = cp.Problem(obj, constraints)
    prob.solve(solver=cp.CLARABEL, verbose=False)
    
    if w.value is None:
        return np.ones(N)/N
    
    # Cleanup numerical noise
    w_final = w.value
    w_final[w_final < 1e-6] = 0
    return w_final / np.sum(w_final)

import numpy as np
import cvxpy as cp

def robust_mvo_box_mean_cov(mu_hat, delta, Sigma_lower, Sigma_upper, lam):
    """
    Solves the Robust Mean-Variance Optimization problem with Box Uncertainty 
    for both the Mean Vector and Covariance Matrix.
    
    Formulation from the provided text:
    Maximize:  w.T @ mu_hat - |w|.T @ delta - (lam/2) * (Worst Case Variance)
    Subject to:
        Sum(w) == 1
        w >= 0
        LMI Constraint for Variance Box
    
    Parameters:
    -----------
    mu_hat      : (N,) array
                  The nominal (estimated) expected returns.
    delta       : (N,) array
                  The half-width of the box uncertainty for returns. 
                  (i.e., true mu is in [mu_hat - delta, mu_hat + delta])
    Sigma_lower : (N, N) array
                  Elementwise lower bound matrix for Sigma.
    Sigma_upper : (N, N) array
                  Elementwise upper bound matrix for Sigma.
    lam         : float
                  Risk aversion parameter (lambda).
                  
    Returns:
    --------
    w : (N,) array, optimal portfolio weights
    """
    N = len(mu_hat)
    
    # --- Variables ---
    # 1. Portfolio weights
    w = cp.Variable(N)
    
    # 2. Dual Variables for the Covariance Box Constraints
    # These correspond to the upper and lower bounds on Sigma.
    # They must be symmetric variables.
    Lam_bar = cp.Variable((N, N), symmetric=True) 
    Lam_und = cp.Variable((N, N), symmetric=True) 

    # --- Objective Components ---
    
    # 1. Robust Mean Return: w.T @ mu_hat - |w|.T @ delta
    # Since we have a constraint w >= 0, |w| is simply w.
    # If we allowed shorting, we would use cp.abs(w).
    robust_mean = w.T @ mu_hat - w.T @ delta
    
    # 2. Robust Variance Term (from Lemma 14.5)
    # The worst-case variance is represented by the dual objective:
    # Tr(Lam_bar * Sigma_upper) - Tr(Lam_und * Sigma_lower)
    wc_variance_proxy = cp.trace(Lam_bar @ Sigma_upper) - cp.trace(Lam_und @ Sigma_lower)
    
    # Maximize Utility: Robust Mean - (lambda/2) * Robust Variance
    # (Note: Standard Mean-Variance usually scales variance by lambda/2)
    obj = cp.Maximize(robust_mean - (lam / 2.0) * wc_variance_proxy)

    # --- Constraints ---
    constraints = [
        # 1. Standard Portfolio Constraints
        cp.sum(w) == 1,
        w >= 0, # Long-only constraint simplifies |w| to w in objective

        # 2. The Linear Matrix Inequality (LMI)
        # This enforces the robust variance consistency via Schur Complement.
        # [ (Lam_bar - Lam_und)   w ]
        # [        w.T            1 ]  >> 0 (PSD)
        cp.bmat([
            [Lam_bar - Lam_und,              cp.reshape(w, (N, 1))],
            [cp.reshape(w, (1, N)),          np.array([[1.0]])]
        ]) >> 0,

        # 3. Dual Constraints (Elementwise Non-Negative)
        # These enforce the box constraints on Sigma.
        Lam_bar >= 0,
        Lam_und >= 0
    ]

    # --- Solve ---
    prob = cp.Problem(obj, constraints)
    
    # Use CLARABEL or SCS (Robust solvers for SDPs)
    try:
        prob.solve(solver=cp.CLARABEL, verbose=False)
    except:
        prob.solve(solver=cp.SCS, verbose=False)

    # --- Output Cleanup ---
    if w.value is None:
        print("Optimization failed or infeasible.")
        return np.ones(N)/N
    
    # Normalize and clean small numerical noise
    w_final = w.value
    w_final[w_final < 1e-6] = 0
    
    # Re-normalize just to be safe
    return w_final / np.sum(w_final)