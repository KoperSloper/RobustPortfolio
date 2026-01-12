import numpy as np
import cvxpy as cp
from scipy.linalg import cholesky, sqrtm

def robust_gmrp_ellipsoidal(mu_hat, S, kappa):
    """
    Solves the Robust GMRP under Ellipsoidal Uncertainty (Lemma 14.1).
    
    Problem:
        Maximize   w.T @ mu_hat - kappa * || S^(1/2) @ w ||_2
        Subject to sum(w) = 1, w >= 0
        
    Parameters:
    -----------
    mu_hat : (N,) array
        The nominal expected returns.
    S      : (N, N) array
        The shape matrix of the ellipsoid (typically Covariance / T).
    kappa  : float
        The size of the uncertainty set (safety factor).
        
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
        
    # Cleanup numerical noise
    w_final = w.value
    w_final[w_final < 1e-6] = 0
    return w_final / np.sum(w_final)


def robust_gmrp_box(mu_hat, delta):
    """
    Solves the Robust GMRP under Box Uncertainty (Lemma 14.2).
    
    Problem:
        Maximize   w.T @ mu_hat - |w|.T @ delta
        Subject to sum(w) = 1, w >= 0
        
    Note:
        Because of the constraint w >= 0, |w| becomes w.
        The problem simplifies to: Maximize w.T @ (mu_hat - delta)
    
    Parameters:
    -----------
    mu_hat : (N,) array
        The nominal expected returns.
    delta  : (N,) array
        The half-width of the uncertainty box (delta >= 0).
        
    Returns:
    --------
    w : (N,) array
        Optimal portfolio weights.
    """
    N = len(mu_hat)
    
    # 1. Define Variables
    w = cp.Variable(N)
    
    # 2. Define Objective
    # We use the explicit absolute value formulation from the lemma text
    # just to be mathematically faithful, though w >= 0 makes it redundant.
    # CVXPY handles cp.abs() automatically by adding internal constraints.
    robust_penalty = cp.abs(w).T @ delta
    
    objective = cp.Maximize(w.T @ mu_hat - robust_penalty)
    
    # 3. Constraints
    constraints = [
        cp.sum(w) == 1,
        w >= 0
    ]
    
    # 4. Solve
    # This is effectively a Linear Program (LP)
    prob = cp.Problem(objective, constraints)
    prob.solve(solver=cp.CLARABEL, verbose=False)
    
    if w.value is None:
        return np.ones(N)/N
    
    # Cleanup numerical noise
    w_final = w.value
    w_final[w_final < 1e-6] = 0
    return w_final / np.sum(w_final)