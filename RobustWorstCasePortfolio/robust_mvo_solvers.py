import numpy as np
import cvxpy as cp
from scipy.linalg import sqrtm, cholesky

def robust_mvo_general_ellipsoid(mu_hat, Sigma_hat, lam, epsilon, kappa, S_mu, P_shape):
    """
    Solves the Robust Mean-Variance Optimization (MVO) using the General SDP formulation.
    Handles custom uncertainty shapes for covariance via the matrix P_shape.

    Problem Formulation:
    --------------------
    Maximize:   (w.T @ mu_hat - kappa * || S_mu^(1/2) @ w ||_2)
                - lambda * (Tr(Sigma_hat @ (W + Z)) + epsilon * || P_shape @ (W + Z) @ P_shape ||_F)

    Subject to: [ W    w ]
                [ w.T  1 ] >> 0
                sum(w) == 1
                w >= 0
                Z >> 0

    Parameters:
    -----------
    mu_hat : (N,) array
        The nominal expected returns vector.
    Sigma_hat : (N, N) array
        The nominal covariance matrix.
    lam : float
        Risk aversion parameter (lambda). 
        Higher value = More conservative (penalizes risk more).
    epsilon : float
        Uncertainty radius for the covariance matrix.
    kappa : float
        Uncertainty radius for the mean vector.
    S_mu : (N, N) array
        Covariance of the mean estimates (typically Sigma / T).
    P_shape : (N, N) symmetric matrix
        Defines the geometry of the covariance uncertainty set.
        - "Regularization" View: P = Sigma^-1/2
        - "Statistical" View:    P = Sigma^1/2
        - "Spherical" View:      P = Identity

    Returns:
    --------
    w : (N,) array
        Optimal portfolio weights.
    """
    N = len(mu_hat)
    
    # Mean Uncertainty Shape (S_mu^1/2)
    try:
        S_mu_sqrt = np.array(cholesky(S_mu))
    except:
        S_mu_sqrt = np.array(sqrtm(S_mu))

    # Variables
    w = cp.Variable(N)
    W = cp.Variable((N, N), symmetric=True)
    Z = cp.Variable((N, N), symmetric=True) 

    # --- 1. Robust Return (Maximize this) ---
    # Nominal Return - Worst Case Mean Penalty
    ret_nominal = w.T @ mu_hat
    ret_penalty = kappa * cp.norm(S_mu_sqrt.T @ w, 2)
    term_return = ret_nominal - ret_penalty
    
    # --- 2. Robust Risk (Minimize this, so subtract in objective) ---
    # Nominal Risk (SDP form)
    risk_nominal = cp.trace(Sigma_hat @ (W + Z))
    # Robust Covariance Penalty
    risk_robust = epsilon * cp.norm(P_shape @ (W + Z) @ P_shape, "fro")
    
    term_risk = risk_nominal + risk_robust
    
    # --- Objective: Utility = Return - lambda * Risk ---
    obj = cp.Maximize(term_return - lam * term_risk)

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
    try:
        prob.solve(solver=cp.CLARABEL, verbose=False)
    except:
        try:
            prob.solve(solver=cp.SCS, verbose=False, eps=1e-4)
        except:
            # Last resort
            prob.solve(verbose=False)

    if w.value is None or prob.status not in ["optimal", "optimal_inaccurate"]:
        # Fallback to naive 1/N if robust fails completely
        return np.ones(N)/N
        
    return np.maximum(w.value, 0) / np.sum(np.maximum(w.value, 0))

def robust_mvo_fast_ellipsoid(mu_hat, Sigma_hat, lam, epsilon, kappa, Q_cov, S_mu):
    """
    Solves the Robust Mean-Variance Optimization (MVO) using the simplified formulation.
    Assumes Z -> 0, reducing the problem size significantly (Vector-only SOCP).

    Problem Formulation:
    --------------------
    Maximize:   (w.T @ mu_hat - kappa * || S_mu^(1/2) @ w ||_2)
                - lambda * (w.T @ Sigma_hat @ w + epsilon * (Robust Penalty))

    Subject to: sum(w) == 1
                w >= 0

    Parameters:
    -----------
    mu_hat : (N,) array
        The nominal expected returns vector.
    Sigma_hat : (N, N) array
        The nominal covariance matrix.
    lam : float
        Risk aversion parameter (lambda).
        Higher value = More conservative.
    epsilon : float
        Uncertainty radius for the covariance matrix.
    kappa : float
        Uncertainty radius for the mean vector.
    Q_cov : (N, N) array
        The shape matrix for covariance uncertainty. The penalty becomes a quadratic form.
    S_mu : (N, N) array
        Covariance of the mean estimates (typically Sigma / T).

    Returns:
    --------
    w : (N,) array
        Optimal portfolio weights.
    """
    N = len(mu_hat)
    
    try:
        S_mu_sqrt = np.array(cholesky(S_mu))
    except:
        S_mu_sqrt = np.array(sqrtm(S_mu))

    # Variables (Vector only)
    w = cp.Variable(N)
    
    # --- 1. Robust Return ---
    ret_nominal = w.T @ mu_hat
    ret_penalty = kappa * cp.norm(S_mu_sqrt.T @ w, 2)
    term_return = ret_nominal - ret_penalty

    # --- 2. Robust Risk ---
    # Nominal Risk
    risk_nominal = cp.quad_form(w, Sigma_hat)
    
    # Robust Penalty
    if np.array_equal(Q_cov, np.eye(N)):
        risk_robust = epsilon * cp.sum_squares(w)
    else:
        risk_robust = epsilon * cp.quad_form(w, Q_cov)
        
    term_risk = risk_nominal + risk_robust
    
    # --- Objective ---
    obj = cp.Maximize(term_return - lam * term_risk)
    
    constraints = [
        cp.sum(w) == 1,
        w >= 0
    ]
    
    prob = cp.Problem(obj, constraints)
    try:
        prob.solve(solver=cp.CLARABEL, verbose=False)
    except:
        prob.solve(solver=cp.SCS, verbose=False)
    
    if w.value is None:
        return np.ones(N)/N
    
    w_final = w.value
    w_final[w_final < 1e-6] = 0
    return w_final / np.sum(w_final)

def robust_mvo_box(mu_hat, delta, Sigma_lower, Sigma_upper, lam):
    """
    Solves the Robust Mean-Variance Optimization (MVO) under Box Uncertainty.
    Fuses the portfolio maximization with the dual minimization of the worst-case variance.

    Problem Formulation:
    --------------------
    Maximize:   (w.T @ mu_hat - |w|.T @ delta) - (lambda / 2) * (Worst-Case Variance)
    Subject to: sum(w) == 1
                w >= 0
                [ (Lam_bar - Lam_und)   w ]
                [        w.T            1 ] >> 0  (LMI for Worst-Case Variance)
                Lam_bar >= 0, Lam_und >= 0

    Parameters:
    -----------
    mu_hat : (N,) array
        The nominal expected returns vector.
    delta : (N,) array
        Half-width of the return uncertainty box (true mu in [mu_hat - delta, mu_hat + delta]).
    Sigma_lower : (N, N) array
        Elementwise lower bound matrix for Sigma.
    Sigma_upper : (N, N) array
        Elementwise upper bound matrix for Sigma.
    lam : float
        Risk aversion parameter (lambda).

    Returns:
    --------
    w : (N,) array
        Optimal portfolio weights.
    """
    N = len(mu_hat)

    w = cp.Variable(N)
    
    Lam_bar = cp.Variable((N, N), symmetric=True) 
    Lam_und = cp.Variable((N, N), symmetric=True) 

    robust_mean = w.T @ mu_hat - w.T @ delta
    
    wc_variance_proxy = cp.trace(Lam_bar @ Sigma_upper) - cp.trace(Lam_und @ Sigma_lower)
    
    obj = cp.Maximize(robust_mean - (lam / 2.0) * wc_variance_proxy)

    constraints = [
        # 1. Standard Portfolio Constraints
        cp.sum(w) == 1,
        w >= 0, # Long-only constraint simplifies |w| to w in objective

        cp.bmat([
            [Lam_bar - Lam_und,              cp.reshape(w, (N, 1))],
            [cp.reshape(w, (1, N)),          np.array([[1.0]])]
        ]) >> 0,

        Lam_bar >= 0,
        Lam_und >= 0
    ]

    prob = cp.Problem(obj, constraints)
    
    try:
        prob.solve(solver=cp.CLARABEL, verbose=False)
    except:
        prob.solve(solver=cp.SCS, verbose=False)

    if w.value is None:
        print("Optimization failed or infeasible.")
        return np.ones(N)/N

    w_final = w.value
    w_final[w_final < 1e-6] = 0

    return w_final / np.sum(w_final)