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