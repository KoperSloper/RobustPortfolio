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
    
    # convert Net Returns to Gross Returns (1 + r) for Log Utility
    R_gross = 1.0 + returns_data
    
    # Objective: Maximize sum of log(portfolio_returns)
    portfolio_gross_ret = R_gross @ w
    objective = cp.Maximize(cp.sum(cp.log(portfolio_gross_ret)))
    
    # Constraints: Long-only, Fully Invested
    constraints = [
        cp.sum(w) == 1,
        w >= 0
    ]
    
    prob = cp.Problem(objective, constraints)
    
    try:
        prob.solve(solver=cp.ECOS, verbose=False)
    except:
        try:
            prob.solve(solver=cp.SCS, verbose=False)
        except:
            return np.ones(N) / N

    if w.value is None:
        return np.ones(N) / N
        
    w_final = w.value
    w_final[w_final < 1e-6] = 0

    if np.sum(w_final) == 0:
        return np.ones(N) / N
        
    return w_final / np.sum(w_final)

def ensemble_growth_optimal_portfolio(returns, n1, n2, n3, n4):
    """
    Algorithm 1: Ensemble Growth Optimal (EGO) Portfolio
    
    Combines Parametric Bootstrapping (Bagging) with Random Subspace 
    method to stabilize the Kelly (Growth Optimal) Criterion.

    Parameters:
    -----------
    returns : np.ndarray
        (T x n) matrix of historical return data (R_l).
    n1 : int
        Number of resamples (simulated histories/outer loop).
    n2 : int
        Size of each resample (length of synthetic history).
    n3 : int
        Number of resampled subsets (inner loop).
    n4 : int
        Size of each subset (number of assets per team).

    Returns:
    --------
    np.ndarray
        (n,) vector of final aggregated portfolio weights.
    """
    T, n = returns.shape
    
    # compute sample estimates from original data
    mu_hat = np.mean(returns, axis=0)
    Sigma_hat = np.cov(returns, rowvar=False)
    
    final_weight_accumulator = np.zeros(n)
    
    for h in range(n1):
        
        # create n2 periods of fake history using the estimated stats
        try:
            # generate (n2 x n) matrix
            synthetic_returns = np.random.multivariate_normal(
                mean=mu_hat, 
                cov=Sigma_hat, 
                size=n2
            )
        except np.linalg.LinAlgError:
            # fallback for non-PSD covariance
            jitter = 1e-6 * np.eye(n)
            synthetic_returns = np.random.multivariate_normal(
                mean=mu_hat, 
                cov=Sigma_hat + jitter, 
                size=n2
            )

        # initialize aggregator for this specific history's basis portfolio
        basis_portfolio_accumulator = np.zeros(n)
        
        for j in range(n3):
            
            # randomly select n4 assets without replacement
            subset_indices = np.random.choice(n, size=n4, replace=False)
            
            # select synthetic data for these indices
            subset_data = synthetic_returns[:, subset_indices]
            
            # compute optimal subset weights
            w_subset = naive_growth_optimal(subset_data)
            
            # construct weights for basis portfolio
            # map small subset weights back to full n-sized vector
            w_full_projection = np.zeros(n)
            w_full_projection[subset_indices] = w_subset
            
            # accumulate
            basis_portfolio_accumulator += w_full_projection
            
        # average the subsets to get the basis portfolio for history 'h'
        omega_k_h = basis_portfolio_accumulator / n3
        
        # add this basis portfolio to the total ensemble
        final_weight_accumulator += omega_k_h

    # aggregate all basis portfolios
    omega_k = final_weight_accumulator / n1
    
    return omega_k / np.sum(omega_k)