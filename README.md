<p align="center">
  <img src="Images/robustportfolio.png" width="500" title="RobustPortfolio Logo">
</p>

# RobustPortfolio

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Status](https://img.shields.io/badge/status-active-success.svg)]()

> **Advanced quantitative tools for constructing stable, performing portfolios using robust optimization and ensemble methods.**

## Overview

Standard portfolio optimization (like Mean-Variance Optimization) is notoriously sensitive to input noise. Slight errors in estimating asset returns or covariance matrices often lead to "error maximization", unstable portfolios that perform poorly out-of-sample.

**RobustPortfolio** is a Python library designed to mitigate these issues based on the books Feng and Palomar (2016) and Palomar (2025). It provides solvers that:
1.  **Incorporate Uncertainty Sets:** Optimizes for the worst-case scenario within a defined uncertainty set.
2.  **Utilize Ensemble Methods:** Applies bagging and subset resampling to stabilize Growth Optimal (Kelly) portfolios.

## Performance Visuals

### 1. Robust vs. Naive
Standard Mean-Variance optimization often produces unstable out-of-sample results. The plots below compare the distribution of **Sharpe Ratios** and **Maximum Drawdowns** for a Robust Worst-Case (RWC) portfolio versus a Naive implementation.

![RobustWorstCase](Images/OOS_RWC.png)

> **Key Observation:**
> * **Sharpe Ratio (Left):** The Robust portfolio (Pink) shifts the distribution of risk-adjusted returns to the right, demonstrating superior performance per unit of risk compared to the Naive approach (Purple).
> * **Drawdown (Right):** The Robust method effectively truncates the "tail risk," resulting in significantly shallower maximum drawdowns.

### 2. Ensemble Methods
Standard growth-optimal portfolios (Kelly) are risky and volatile. By using **Ensemble Resampling**, we smooth out the "extreme" bets. The plots below show the **distribution** of out-of-sample growth rates and maximum drawdowns.

![PortfolioResampling](Images/OOS_PR.png)

> **Key Observation:**
> * **Growth Rate (Left):** The ensemble method (Green) produces a tighter, more predictable distribution of growth rates compared to the "fat-tailed" and unstable Naive Kelly approach (Orange).
> * **Drawdown (Right):** The ensemble approach successfully shifts the risk distribution, reducing the probability of catastrophic drawdowns.

## Project Structure

```text
RobustPortfolio/
├── Images/                          # Logos and plot screenshots
├── RobustWorstCasePortfolio/        # Source code (The Package)
│   ├── robust_gmrp_solvers.py
│   ├── robust_gmvp_solvers.py
│   ├── robust_mvo_solvers.py
│   └── robust_solvers_factors.py
├── PortfolioResampling/  
|   ├── EnsembleGrowthOptimal.py
│   └── SubsetResampling.py
├── HelperFunctions/
│   └── Functions.py
├── example_PR.ipynb                # Example   
├── example_RWCP.ipynb              # Example
├── setup.py                        # Installation config
└── README.md
```

## Installation

To install the package in use:

```bash
git clone https://github.com/KoperSloper/RobustPortfolio.git
cd RobustPortfolio
pip install -e .
```

## Quick Start

Below is an example of running the Robust Mean-Variance Optimizer with ellipsoidal uncertainty sets.

```python
import numpy as np
from RobustWorstCasePortfolio.robust_mvo_solvers import robust_mvo_general_ellipsoid

# 1. Define Input Data (3 Assets)
T = 52  # e.g., Weekly observations
mu_hat = np.array([0.12, 0.10, 0.07])
Sigma_hat = np.array([[0.0064, 0.0008, 0.0011],
                      [0.0008, 0.0025, 0.0014],
                      [0.0011, 0.0014, 0.0040]])

# 2. Set Robust Parameters
lam = 2.0        # Risk Aversion
epsilon = 0.1    # Size of Covariance uncertainty set
kappa = 0.1      # Size of Expected Returns uncertainty set

# 3. Define Shape Matrices
# S_mu shapes the uncertainty around returns (typically scaled by sample size)
S_mu = Sigma_hat / T 
P_shape = np.eye(3)

# 4. Solve
weights = robust_mvo_general_ellipsoid(
    mu_hat, Sigma_hat, lam, epsilon, kappa, S_mu, P_shape
)

print("Optimal Robust Weights:")
print(weights)
```

## References

This library implements algorithms and concepts discussed in the following texts:

1.  **Feng, Y., & Palomar, D. P. (2016).** *A Signal Processing Perspective on Financial Engineering.* Foundations and Trends® in Signal Processing.
2.  **Palomar, D. P. (2025).** *Portfolio Optimization Book* (2025).
3.  **Shen, W., & Wang, J. (2017).** Portfolio Selection via Subset Resampling. [cite_start]*Proceedings of the Thirty-First AAAI Conference on Artificial Intelligence (AAAI-17).
4.  **Shen, W., Wang, B., Pu, J., & Wang, J. (2019).** The Kelly Growth Optimal Portfolio with Ensemble Learning. [cite_start]*The Thirty-Third AAAI Conference on Artificial Intelligence (AAAI-19).

## Contact

**Mail:** ton.vossen@outlook.com
