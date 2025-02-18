import numpy as np
from scipy.optimize import minimize

def mean_variance_optimization(expected_returns, cov_matrix, target_return):
    """
    Perform mean-variance optimization to find optimal portfolio weights.
    
    Parameters:
    - expected_returns: Array of expected returns for each pair.
    - cov_matrix: Covariance matrix of pair returns.
    - target_return: Desired portfolio return.
    
    Returns:
    - Optimal weights for each pair.
    """
    num_assets = len(expected_returns)
    
    # Objective function: Minimize portfolio variance
    def portfolio_variance(weights):
        return np.dot(weights.T, np.dot(cov_matrix, weights))
    
    # Constraints
    constraints = [
        {"type": "eq", "fun": lambda weights: np.sum(weights) - 1},  # Weights sum to 1
        {"type": "eq", "fun": lambda weights: np.dot(weights, expected_returns) - target_return}  # Achieve target return
    ]
    
    # Bounds: Weights must be between 0 and 1 (no short selling)
    bounds = [(0, 1) for _ in range(num_assets)]
    
    # Initial guess: Equal weights
    initial_weights = np.ones(num_assets) / num_assets
    
    # Perform optimization
    result = minimize(portfolio_variance, initial_weights, method="SLSQP", bounds=bounds, constraints=constraints)
    
    if not result.success:
        raise ValueError("Optimization failed: " + result.message)
    
    return result.x
