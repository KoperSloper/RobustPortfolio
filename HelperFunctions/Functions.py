import numpy as np

def max_drawdown(return_series):
    """Max peak-to-trough decline."""
    comp_ret = (1 + return_series).cumprod()
    peak = np.maximum.accumulate(comp_ret)
    drawdown = (comp_ret - peak) / peak
    return drawdown.min()

def sharpe_ratio(return_series, rf=0.0):
    """Annualized Sharpe Ratio (assuming weekly data)."""
    mean = np.mean(return_series) - rf
    std = np.std(return_series)
    if std < 1e-9: return 0.0
    return (mean / std) * np.sqrt(52)

def geometric_growth_rate(returns, annualized_factor=52):
    """
    Calculates the annualized geometric growth rate (Log-Growth).
    """
    T = len(returns)
    # G = (Product(1 + r))^(1/T) - 1  ~=  Mean - Var/2
    compounded_growth = np.prod(1 + returns) ** (1 / T) - 1
    return compounded_growth * annualized_factor