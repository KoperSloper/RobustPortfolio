import numpy as np
import cvxpy as cp

def naive_mvo(mu, Sigma, lam):
    """
    Standard MVO (Utility Maximization) with Long-Only constraints.
    Helper function for the subset optimization.
    """
    N = len(mu)
    w = cp.Variable(N)
    
    # Utility = Returns - Lambda * Risk
    ret = w.T @ mu
    risk = cp.quad_form(w, Sigma)
    obj = cp.Maximize(ret - lam * risk)
    
    constraints = [
        cp.sum(w) == 1,
        w >= 0
    ]
    
    prob = cp.Problem(obj, constraints)
    try:
        # Try primary solver
        prob.solve(solver=cp.CLARABEL, verbose=False)
    except:
        # Fallback solver
        prob.solve(solver=cp.SCS, verbose=False)
    
    # Handle solver failure cases
    if w.value is None: 
        return np.ones(N) / N
    
    w_final = w.value
    # Clean numerical noise
    w_final[w_final < 1e-6] = 0
    
    # Safety check for all-zero vector before division
    if np.sum(w_final) == 0:
        return np.ones(N) / N
        
    return w_final / np.sum(w_final)

def subset_resampling_mvo(returns, s, b, lam):
    """
    Computes the Subset Resampling (SSR) Portfolio.
    
    This algorithm alleviates estimation error by aggregating portfolios 
    constructed from random subsets of assets, rather than optimizing 
    the entire universe at once.

    Parameters:
    -----------
    returns : np.ndarray
        (T x N) matrix of historical returns, where T is time and N is assets.
    s : int
        Number of subsets (resamples) to generate.
    b : int
        Subset size (number of assets in each random subset).
    lam : float
        Risk aversion parameter for the MVO utility function.

    Returns:
    --------
    np.ndarray
        (N,) vector of optimized portfolio weights.
    """
    T, N = returns.shape
    
    # Initialize the accumulator for the final averaged weights
    cumulative_weights = np.zeros(N)

    for j in range(s):
        # randomly select 'b' assets without replacement
        subset_indices = np.random.choice(N, size=b, replace=False)
        
        # select returns for the chosen subset
        subset_returns = returns[:, subset_indices] # Shape (T, b)
        
        mu_sub = np.mean(subset_returns, axis=0)
        Sigma_sub = np.cov(subset_returns, rowvar=False)
        
        # optimal weights for the subset
        w_sub = naive_mvo(mu_sub, Sigma_sub, lam)
        
        # assets not in the subset get a weight of 0.
        w_full_projection = np.zeros(N)
        w_full_projection[subset_indices] = w_sub

        cumulative_weights += w_full_projection

    avg_weights = cumulative_weights / s

    avg_weights[avg_weights < 1e-6] = 0
    return avg_weights / np.sum(avg_weights)