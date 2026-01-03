import numpy as np
import cvxpy as cp
from scipy.linalg import cholesky, sqrtm

def robust_gmrp_ellipsoidal(mu_hat, S, kappa):
    """
    Solves the Robust Global Maximum Return Portfolio (GMRP) under Ellipsoidal Uncertainty.

    Problem Formulation:
    --------------------
    Maximize:   w.T @ mu_hat - kappa * || S^(1/2) @ w ||_2
    Subject to: sum(w) == 1
                w >= 0

    Parameters:
    -----------
    mu_hat : (N,) array
        The nominal expected returns vector.
    S : (N, N) array
        The shape matrix of the uncertainty ellipsoid (typically Covariance / T).
    kappa : float
        The safety factor determining the size of the uncertainty set.

    Returns:
    --------
    w : (N,) array
        Optimal portfolio weights.
    """
    N = len(mu_hat)
    
    try:
        S_factor = cholesky(S) 
    except:
        S_factor = sqrtm(S)

    w = cp.Variable(N)
    
    # Maximize: Nominal Return - Robust Penalty
    robust_penalty = kappa * cp.norm(S_factor @ w, 2)
    
    objective = cp.Maximize(w.T @ mu_hat - robust_penalty)

    constraints = [
        cp.sum(w) == 1,
        w >= 0
    ]

    prob = cp.Problem(objective, constraints)
    prob.solve(solver=cp.CLARABEL, verbose=False) 
    
    if w.value is None:
        return np.ones(N)/N

    w_final = w.value
    w_final[w_final < 1e-6] = 0
    return w_final / np.sum(w_final)


def robust_gmrp_box(mu_hat, delta):
    """
    Solves the Robust Global Maximum Return Portfolio (GMRP) under Box Uncertainty.

    Problem Formulation:
    --------------------
    Maximize:   w.T @ mu_hat - |w|.T @ delta
    Subject to: sum(w) == 1
                w >= 0
    
    Note: Since w >= 0, |w| simplifies to w, reducing this to a linear program
    maximizing w.T @ (mu_hat - delta).

    Parameters:
    -----------
    mu_hat : (N,) array
        The nominal expected returns vector.
    delta : (N,) array
        The half-width of the uncertainty box (must be >= 0).

    Returns:
    --------
    w : (N,) array
        Optimal portfolio weights.
    """
    N = len(mu_hat)

    w = cp.Variable(N)
    
    robust_penalty = cp.abs(w).T @ delta
    
    objective = cp.Maximize(w.T @ mu_hat - robust_penalty)
    
    constraints = [
        cp.sum(w) == 1,
        w >= 0
    ]
    
    prob = cp.Problem(objective, constraints)
    prob.solve(solver=cp.CLARABEL, verbose=False)
    
    if w.value is None:
        return np.ones(N)/N

    w_final = w.value
    w_final[w_final < 1e-6] = 0
    return w_final / np.sum(w_final)