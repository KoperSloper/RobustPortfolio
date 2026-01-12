import numpy as np
import cvxpy as cp

def naive_growth_optimal(returns_data):
    """
    Solves the Growth Optimal Portfolio (Kelly Criterion) problem.
    
    Parameters:
    -----------
    returns_data : np.ndarray
        (T x N) matrix of returns for the specific subset/period.
        Assumes Net Returns.
        
    Returns:
    --------
    np.ndarray : (N,) vector of optimal weights.
    """
    T, N = returns_data.shape
    w = cp.Variable(N)
    
    # Convert Net Returns to Gross Returns (1 + r) for Log Utility
    # We add 1.0 to ensure we are taking the log of wealth, not net rate.
    # We use cp.hstack to handle the matrix multiplication correctly in CVXPY
    R_gross = 1.0 + returns_data
    
    # Objective: Maximize sum of log(portfolio_returns)
    # This is equivalent to maximizing Geometric Mean Return
    portfolio_gross_ret = R_gross @ w
    objective = cp.Maximize(cp.sum(cp.log(portfolio_gross_ret)))
    
    # Constraints: Long-only, Fully Invested
    constraints = [
        cp.sum(w) == 1,
        w >= 0
    ]
    
    prob = cp.Problem(objective, constraints)
    
    try:
        # ECOS is often better for Log-Cone problems than OSQP
        prob.solve(solver=cp.ECOS, verbose=False)
    except:
        try:
            prob.solve(solver=cp.SCS, verbose=False)
        except:
            # Fallback if solvers fail (rare with valid data)
            return np.ones(N) / N

    if w.value is None:
        return np.ones(N) / N
        
    w_final = w.value
    w_final[w_final < 1e-6] = 0
    
    # Safety check for sum
    if np.sum(w_final) == 0:
        return np.ones(N) / N
        
    return w_final / np.sum(w_final)