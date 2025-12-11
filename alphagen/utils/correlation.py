import torch
from torch import Tensor

from alphagen.utils.pytorch_utils import masked_mean_std


def _mask_either_nan(x: Tensor, y: Tensor, fill_with: float = torch.nan):
    x = x.clone()                       # [days, stocks]
    y = y.clone()                       # [days, stocks]
    nan_mask = x.isnan() | y.isnan()
    # nan_mask = (~torch.isfinite(x)) | (~torch.isfinite(y))
    x[nan_mask] = fill_with
    y[nan_mask] = fill_with
    n = (~nan_mask).sum(dim=1)
    return x, y, n, nan_mask


def _rank_data(x: Tensor, nan_mask: Tensor) -> Tensor:
    rank = x.argsort().argsort().float()            # [d, s]
    eq = x[:, None] == x[:, :, None]                # [d, s, s]
    eq = eq / eq.sum(dim=2, keepdim=True)           # [d, s, s]
    rank = (eq @ rank[:, :, None]).squeeze(dim=2)
    rank[nan_mask] = 0
    return rank                                     # [d, s]


# def _rank_data(x: Tensor, nan_mask: Tensor) -> Tensor:
#     x = x.clone()
#     x[nan_mask] = float('nan')
#     x = x.argsort().float()
#     x[nan_mask] = float('nan')
#     rank = x.argsort().float()
#     # rank = x.argsort().argsort().float()            # [d, s]


#     # eq = x[:, None] == x[:, :, None]                # [d, s, s]
#     # eq = eq / eq.sum(dim=2, keepdim=True)           # [d, s, s]

#     # rank = (eq @ rank[:, :, None]).squeeze(dim=2)
#     rank[nan_mask] = 0
#     return rank                                     # [d, s]
    

def _batch_pearsonr_given_mask(
    x: Tensor, y: Tensor,
    n: Tensor, mask: Tensor
) -> Tensor:
    x_mean, x_std = masked_mean_std(x, n, mask)
    y_mean, y_std = masked_mean_std(y, n, mask)
    cov = (x * y).sum(dim=1) / n - x_mean * y_mean
    stdmul = x_std * y_std
    stdmul[(x_std < 1e-3) | (y_std < 1e-3)] = 1
    corrs = cov / stdmul
    return corrs

def _batch_ret_given_mask(
    x: Tensor, y: Tensor,
    n: Tensor, mask: Tensor
) -> Tensor:
    x_mean, x_std = masked_mean_std(x, n, mask)
    y_mean, y_std = masked_mean_std(y, n, mask)
    cov = (x * y).sum(dim=1) / n - x_mean * y_mean
    stdmul = x_std 
    stdmul[(x_std < 1e-3) | (y_std < 1e-3)] = 1
    corrs = cov / (stdmul**2)
    return corrs

def batch_spearmanr(x: Tensor, y: Tensor) -> Tensor:
    x, y, n, nan_mask = _mask_either_nan(x, y)
    rx = _rank_data(x, nan_mask)
    ry = _rank_data(y, nan_mask)
    return _batch_pearsonr_given_mask(rx, ry, n, nan_mask)


def batch_pearsonr(x: Tensor, y: Tensor) -> Tensor:
    res =  _batch_pearsonr_given_mask(*_mask_either_nan(x, y, fill_with=0.))
    # fillna
    res[res.isnan()] = 0
    return res

def batch_ret(x:Tensor,y:Tensor)->Tensor:
    return _batch_ret_given_mask(*_mask_either_nan(x, y, fill_with=0.))

def _mask_either_nan_y_only(x: Tensor, y: Tensor, fill_with: float = torch.nan):
    x = x.clone()                       # [days, stocks]
    y = y.clone()                       # [days, stocks]
    nan_mask = y.isnan()
    # nan_mask = ~torch.isfinite(y)
    x[nan_mask] = fill_with
    y[nan_mask] = fill_with
    n = (~nan_mask).sum(dim=1)
    return x, y, n, nan_mask
def batch_pearsonr_full_y(x:Tensor,y:Tensor)->Tensor:
    x,y,n,nan_mask = _mask_either_nan_y_only(x,y,fill_with=0.)
    return _batch_pearsonr_given_mask(x,y,n,nan_mask)

def batch_sharpe_ratio(ret_series: Tensor, risk_free_rate: float = 0.0) -> Tensor:
    """
    Calculate Sharpe Ratio for return series.
    
    Args:
        ret_series: Tensor of shape [days] - daily return series
        risk_free_rate: Risk-free rate (annualized), default 0.0
    
    Returns:
        Sharpe ratio as a scalar tensor
    """
    # Convert to daily risk-free rate (assuming 252 trading days per year)
    daily_rf = risk_free_rate / 252.0
    
    # Calculate excess returns
    excess_returns = ret_series - daily_rf
    
    # Calculate mean and std of excess returns
    mean_excess = excess_returns.mean()
    std_excess = excess_returns.std()
    
    # Handle division by zero
    if std_excess < 1e-8:
        return torch.tensor(0.0, device=ret_series.device)
    
    # Annualize Sharpe ratio
    sharpe = mean_excess / std_excess * (252.0 ** 0.5)
    
    return sharpe

def batch_max_drawdown(ret_series: Tensor) -> Tensor:
    """
    Calculate Maximum Drawdown for return series.
    
    Args:
        ret_series: Tensor of shape [days] - daily return series
    
    Returns:
        Maximum drawdown as a scalar tensor (positive value)
    """
    # Calculate cumulative returns
    cum_returns = 1+ret_series.cumsum(dim=0)
    # Calculate running maximum using cummax
    running_max = torch.cummax(cum_returns, dim=0)[0]
    
    # Calculate drawdown at each point
    drawdown = (cum_returns - running_max) / running_max
    
    # Return maximum drawdown (as positive value)
    max_dd = -drawdown.min()
    
    return max_dd