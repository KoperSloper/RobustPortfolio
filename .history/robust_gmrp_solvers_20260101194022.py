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
    
    # 1. Compute the factorization of S to represent S^(1/2).
    # We want a matrix L such that || L @ w ||_2^2 = w.T @ S @ w.
    # Scipy cholesky returns U such that U.T @ U = S. 
    # Therefore, || U @ w ||_2^2 = w.T @ U.T @ U @ w = w.T @ S @ w.
    try:
        # Standard Cholesky (S must be PD)
        S_factor = cholesky(S) # Returns Upper triangular
    except:
        # Fallback for PSD matrices (slightly slower but robust)
        S_factor = sqrtm(S)

    # 2. Define Variables
    w = cp.Variable(N)
    
    # 3. Define Objective
    # Maximize: Nominal Return - Robust Penalty
    # The term || S^(1/2) w ||_2 is equivalent to cp.norm(S_factor @ w, 2)
    robust_penalty = kappa * cp.norm(S_factor @ w, 2)
    
    objective = cp.Maximize(w.T @ mu_hat - robust_penalty)
    
    # 4. Constraints
    constraints = [
        cp.sum(w) == 1,
        w >= 0
    ]
    
    # 5. Solve
    prob = cp.Problem(objective, constraints)
    prob.solve(solver=cp.CLARABEL, verbose=False) # CLARABEL handles SOCPs well
    
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


# --- Example Usage ---
if __name__ == "__main__":
    # 1. Setup Dummy Data
    N = 4
    mu_hat = np.array([0.05, 0.08, 0.12, 0.06])
    
    # Covariance Matrix for Ellipsoidal
    tmp = np.random.randn(N, N)
    Sigma = tmp.T @ tmp + np.eye(N)*0.01
    S = Sigma / 100.0 # e.g., scaled by 1/T
    
    # Box parameters
    delta = np.array([0.02, 0.03, 0.04, 0.02])
    
    # 2. Solve Ellipsoidal Case
    # kappa = 1.0 (1 std dev confidence roughly)
    w_ellip = robust_gmrp_ellipsoidal(mu_hat, S, kappa=1.0)
    print("Ellipsoidal Robust Weights:", np.round(w_ellip, 3))
    
    # 3. Solve Box Case
    w_box = robust_gmrp_box(mu_hat, delta)
    print("Box Robust Weights:        ", np.round(w_box, 3))