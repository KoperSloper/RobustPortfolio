import cvxpy as cp
import numpy as np

def robust_gmvp_factor_model(Pi_hat, D_upper_diag, delta):
    """
    Solves the Robust Global Minimum Variance Portfolio (GMVP) using a Factor Model.
    
    Problem Formulation:
    --------------------
    Minimize:   ( || Pi_hat @ w ||_2 + delta * || w ||_2 )^2  +  w.T @ diag(D_upper) @ w
    Subject to: sum(w) == 1
                w >= 0
    
    This corresponds to the worst-case variance where:
      - Factor loadings (Pi) lie in a spherical uncertainty set with radius 'delta'.
      - Residual variances (D) lie in a box set with upper bound 'D_upper'.
    
    Parameters:
    -----------
    Pi_hat : (K, N) array
        The estimated factor loading matrix (K factors, N assets).
    D_upper_diag : (N,) array
        The vector of upper bounds for the residual variances.
    delta : float
        The uncertainty radius for the factor loadings.
        
    Returns:
    --------
    w : (N,) array
        Optimal portfolio weights.
    """
    K, N = Pi_hat.shape

    w = cp.Variable(N)
    
    # Worst-Case Factor Risk 
    nominal_factor_risk = cp.norm(Pi_hat @ w, 2)
    uncertainty_penalty = delta * cp.norm(w, 2)
    total_factor_std = nominal_factor_risk + uncertainty_penalty
    
    # Worst-Case Idiosyncratic Risk
    D_sqrt = np.sqrt(D_upper_diag)
    idiosyncratic_variance = cp.sum_squares(cp.multiply(D_sqrt, w))
    
    # Minimize: (Factor_Std)^2 + Idiosyncratic_Var
    obj = cp.Minimize(cp.square(total_factor_std) + idiosyncratic_variance)
    
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
        print("Optimization Failed.")
        return np.ones(N)/N
        
    w_final = w.value
    w_final[w_final < 1e-6] = 0
    return w_final / np.sum(w_final)

def robust_mvo_factor_model(mu, Pi_hat, D_upper_diag, delta, lambda_reg):
    """
    Solves the Robust Mean-Variance Optimization (MVO) using a Factor Model.
    
    Mathematical Formulation:
    -------------------------
    Maximize:   w^T mu - lambda * ( w^T D_bar w + y^2 )
    Subject to: sum(w) == 1
                w >= 0
                || Pi_hat @ w ||_2 + delta * || w ||_2 <= y
    
    Parameters:
    -----------
    mu : (N,) array
        Expected returns vector.
    Pi_hat : (K, N) array
        Estimated factor loading matrix (K factors, N assets).
        Note: Checks dimensions to ensure it aligns with w.
    D_upper_diag : (N,) array
        Upper bound of residual variances (diagonal elements).
    delta : float
        Uncertainty radius for the factor loadings (delta_Pi).
    lambda_reg : float
        Risk aversion parameter (lambda). 
        Higher values prioritize low risk over high returns.

    Returns:
    --------
    w : (N,) array
        Optimal portfolio weights.
    """
    K, N = Pi_hat.shape

    w = cp.Variable(N)
    y = cp.Variable(1) # Auxiliary variable for Factor Risk bound
    
    # Return Term: w^T mu
    expected_return = mu @ w
    
    # Residual Risk Term: w^T D_bar w
    D_sqrt = np.sqrt(D_upper_diag)
    residual_risk = cp.sum_squares(cp.multiply(D_sqrt, w))

    factor_risk_bound = cp.square(y)
    
    # Maximize: Returns - Lambda * (Residual_Risk + Factor_Risk)
    objective = cp.Maximize(expected_return - lambda_reg * (residual_risk + factor_risk_bound))
    
    constraints = [
        cp.sum(w) == 1,
        w >= 0,

        cp.norm(Pi_hat @ w, 2) + delta * cp.norm(w, 2) <= y
    ]

    prob = cp.Problem(objective, constraints)
    
    try:
        prob.solve(solver=cp.CLARABEL, verbose=False)
    except:
        prob.solve(solver=cp.SCS, verbose=False)
        
    if w.value is None:
        print("Optimization Failed.")
        return np.zeros(N)

    w_final = w.value
    w_final[w_final < 1e-6] = 0
    return w_final / np.sum(w_final)