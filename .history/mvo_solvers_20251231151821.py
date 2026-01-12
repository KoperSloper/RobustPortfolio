import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt
from scipy.linalg import sqrtm
from scipy.stats import chi2

def solve_robust_mvo_explicit(mu_hat, Sigma_hat, lam, epsilon, kappa, S_mu):
    """
    Solves the Robust MVO problem using the Explicit SDP formulation.
    
    Variables:
        w (N,): Portfolio weights.
        W (N,N): Symmetric matrix approximating w*w.T.
        Z (N,N): Symmetric auxiliary matrix for uncertainty.
        
    Objective:
        Minimize: Trace(Sigma*(W+Z)) + epsilon*||P(W+Z)P|| - lambda*Return + lambda*kappa*||S_mu^1/2 w||
    """
    N = len(mu_hat)
    
    # A. Covariance Uncertainty Shape (P = Sigma^-1/2)
    vals, vecs = np.linalg.eigh(Sigma_hat)
    vals = np.maximum(vals, 1e-6)
    inv_sqrt_vals = 1.0 / np.sqrt(vals)
    P = vecs @ np.diag(inv_sqrt_vals) @ vecs.T
    
    # B. Mean Uncertainty Shape (S_mu^1/2)
    try:
        S_mu_sqrt = np.linalg.cholesky(S_mu)
    except:
        S_mu_sqrt = sqrtm(S_mu)

    # variables
    w = cp.Variable(N)
    W = cp.Variable((N, N), symmetric=True)
    Z = cp.Variable((N, N), symmetric=True)
    
    # objective components
    # Tr( Sigma * (W + Z) )
    term_trace = cp.trace(Sigma_hat @ (W + Z))
    
    # epsilon * || P(W + Z)P ||_fro
    term_norm_cov = epsilon * cp.norm(P @ (W + Z) @ P, "fro")
    
    term_ret_est = -lam * (w.T @ mu_hat)

    # lambda * kappa * || S_mu^1/2 * w ||_2
    term_ret_penalty = lam * kappa * cp.norm(S_mu_sqrt.T @ w, 2)
    
    obj = cp.Minimize(term_trace + term_norm_cov + term_ret_est + term_ret_penalty)

    constraints = [
        # 1. Standard Portfolio Constraints
        cp.sum(w) == 1,
        w >= 0,
        
        # 2. Schur Complement for W (Relaxation of W = ww^T)
        # [ W   w ]
        # [ w.T 1 ]  >> 0
        cp.bmat([
        [W, cp.reshape(w, (N, 1), order='C')],
        [cp.reshape(w, (1, N), order='C'), np.array([[1.0]])]
    ]) >> 0,
        
        # 3. Non-negativity for Z
        Z >> 0
    ]

    prob = cp.Problem(obj, constraints)
    
    prob.solve(solver=cp.SCS, verbose=False, eps=1e-4, acceleration_lookback=0)

    if w.value is None:
        return np.ones(N)/N
    
    # Clean weights
    w_final = w.value
    w_final[w_final < 1e-5] = 0
    return w_final / np.sum(w_final)

if __name__ == "__main__":
    np.random.seed(40)
    
    # Settings
    N_ASSETS = 40
    N_OBS = 252       # 1 year of daily data
    LAMBDA = 4.0      # Risk Aversion
    N_SIMS = 1
    
    print(f"Running Simulation: {N_ASSETS} Assets, {N_OBS} Observations per simulation.")
    print("Comparing Naive MVO vs. Explicit Robust MVO (Option A: Matrix Scaling)...\n")

    # 1. True World Parameters
    true_mu = np.linspace(0.05, 0.25, N_ASSETS)
    
    # Generate True Covariance
    vols = np.linspace(0.10, 0.60, N_ASSETS)
    corr = np.random.uniform(0.2, 0.6, (N_ASSETS, N_ASSETS))
    np.fill_diagonal(corr, 1.0)
    corr = (corr + corr.T)/2
    D = np.diag(vols)
    true_sigma = D @ corr @ D
    
    vals, vecs = np.linalg.eigh(true_sigma)
    vals = np.maximum(vals, 1e-6) # Clip negative eigenvalues to 0 or small positive
    true_sigma = vecs @ np.diag(vals) @ vecs.T

    naive_sharpes = []
    robust_sharpes = []

    for i in range(N_SIMS):
        # 2. Generate Data
        X_sample = np.random.multivariate_normal(true_mu, true_sigma, N_OBS)
        mu_hat = np.mean(X_sample, axis=0)
        sigma_hat = np.cov(X_sample, rowvar=False)
        
        # Enforce PSD on sigma_hat
        val, vec = np.linalg.eigh(sigma_hat)
        sigma_hat = vec @ np.diag(np.maximum(val, 1e-5)) @ vec.T

        w_nv = cp.Variable(N_ASSETS)
        risk_nv = cp.quad_form(w_nv, sigma_hat)
        ret_nv = w_nv.T @ mu_hat
        cp.Problem(cp.Minimize(risk_nv - LAMBDA*ret_nv), 
                   [cp.sum(w_nv)==1, w_nv>=0]).solve(solver=cp.OSQP, verbose=False)
        weights_naive = w_nv.value
        if weights_naive is None: weights_naive = np.ones(N_ASSETS)/N_ASSETS

        kappa_raw = np.sqrt(chi2.ppf(0.90, df=N_ASSETS))
        
        # 2. Matrix (Scaled by T)
        # We divide by T here.
        S_mu_scaled = sigma_hat / N_OBS
        
        # 3. Epsilon (Covariance Heuristic)
        epsilon_cal = 0.5 * np.sqrt(N_ASSETS / N_OBS)
        
        weights_robust = solve_robust_mvo_explicit(
            mu_hat, sigma_hat, LAMBDA, epsilon_cal, kappa_raw, S_mu_scaled
        )

        # 4. Evaluate on TRUE Parameters
        r_n = weights_naive @ true_mu
        v_n = np.sqrt(weights_naive @ true_sigma @ weights_naive)
        naive_sharpes.append(r_n / v_n)
        
        r_r = weights_robust @ true_mu
        v_r = np.sqrt(weights_robust @ true_sigma @ weights_robust)
        robust_sharpes.append(r_r / v_r)

    # 5. Results
    avg_naive = np.mean(naive_sharpes)
    avg_robust = np.mean(robust_sharpes)
    
    print("-" * 50)
    print(f"{'Method':<25} | {'True Sharpe Ratio':<15}")
    print("-" * 50)
    print(f"{'Naive MVO':<25} | {avg_naive:.4f}")
    print(f"{'Robust MVO (Option A)':<25} | {avg_robust:.4f}")
    print("-" * 50)
    print(f"Improvement: {((avg_robust-avg_naive)/avg_naive)*100:.2f}%")

    # Plot Weights (Last Simulation)
    plt.figure(figsize=(12,5))
    plt.subplot(1,2,1)
    plt.bar(range(N_ASSETS), weights_naive, color='blue', alpha=0.7)
    plt.title('Naive MVO Weights')
    plt.xlabel('Asset Index')
    plt.ylabel('Weight')
    
    plt.subplot(1,2,2)
    plt.bar(range(N_ASSETS), weights_robust, color='green', alpha=0.7)
    plt.title('Robust MVO Weights')
    plt.xlabel('Asset Index')
    plt.show()