import numpy as np
from scipy.stats import norm, t
from scipy.optimize import minimize
from scipy.special import kv

def validate_data(u, v):
    """Validate ranked returns to ensure they are within [0, 1]."""
    u = np.array(u)
    v = np.array(v)
    # Remove NaNs and values outside [0, 1]
    mask = (u > 0) & (u < 1) & (v > 0) & (v < 1)
    return u[mask], v[mask]

def gaussian_copula_log_likelihood(params, u, v):
    rho = params[0]
    z_u = norm.ppf(u)
    z_v = norm.ppf(v)
    if abs(rho) >= 1:
        return np.inf  # Avoid invalid correlation values
    try:
        log_likelihood = (
            -0.5 * np.log(1 - rho**2)
            - (rho * z_u * z_v - 0.5 * rho**2 * (z_u**2 + z_v**2)) / (1 - rho**2)
        )
        return -np.sum(log_likelihood)  # Negative for minimization
    except (ValueError, RuntimeWarning):
        return np.inf

def student_t_copula_log_likelihood(params, u, v):
    rho, nu = params
    z_u = t.ppf(u, df=nu)
    z_v = t.ppf(v, df=nu)
    if abs(rho) >= 1 or nu <= 0:
        return np.inf  # Avoid invalid parameters
    try:
        log_likelihood = (
            np.log(nu / (nu - 2))
            + np.log(kv((nu + 2) / 2, np.sqrt((nu + z_u**2) * (nu + z_v**2) / (1 - rho**2))))
            - np.log(np.sqrt(1 - rho**2))
        )
        return -np.sum(log_likelihood)  # Negative for minimization
    except (ValueError, RuntimeWarning):
        return np.inf

def select_best_copula(u, v):
    """
    Select the best-fitting copula model (Gaussian or Student-t).
    """
    u, v = validate_data(u, v)

    # Fit Gaussian copula
    initial_params_gaussian = [0.5]  # Initial guess for rho
    result_gaussian = minimize(gaussian_copula_log_likelihood, initial_params_gaussian, args=(u, v), bounds=[(-0.99, 0.99)])
    aic_gaussian = 2 * result_gaussian.fun + 2 * len(initial_params_gaussian)

    # Fit Student-t copula
    initial_params_student_t = [0.5, 5]  # Initial guesses for rho and nu
    result_student_t = minimize(student_t_copula_log_likelihood, initial_params_student_t, args=(u, v), bounds=[(-0.99, 0.99), (2.01, None)])
    aic_student_t = 2 * result_student_t.fun + 2 * len(initial_params_student_t)

    # Select the copula with the lower AIC
    if aic_gaussian < aic_student_t:
        return {
            "type": "gaussian",
            "params": result_gaussian.x,
            "aic": aic_gaussian
        }
    else:
        return {
            "type": "student_t",
            "params": result_student_t.x,
            "aic": aic_student_t
        }

def calculate_tail_dependence(u, v, quantile=0.05):
    """
    Calculate lower and upper tail dependence for Gaussian and Student-t copulas.
    """
    u_sorted = np.sort(u)
    v_sorted = np.sort(v)
    lower_quantile = np.quantile(u_sorted, quantile)
    upper_quantile = np.quantile(u_sorted, 1 - quantile)

    # Lower tail dependence
    lower_tail = np.mean(v_sorted[u_sorted <= lower_quantile])
    # Upper tail dependence
    upper_tail = np.mean(v_sorted[u_sorted >= upper_quantile])

    return lower_tail, upper_tail