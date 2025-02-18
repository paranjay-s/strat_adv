
from config_strategy_api import z_score_window
from statsmodels.tsa.stattools import coint
import statsmodels.api as sm
import pandas as pd
import numpy as np
import math
 
 
# Extract close prices into a list
def extract_close_prices(prices):
    close_prices = []
    for price_values in prices:
        if math.isnan(float(price_values[4])):
            return []
        close_prices.append(float(price_values[4]))
 
 
    return close_prices

import math

# def extract_close_prices(prices):
#     """
#     Extract close prices into a list.
#     Handles both list-of-lists and list-of-dictionaries formats.
#     """
#     close_prices = []
    
#     for price_values in prices:
#         try:
#             # Case 1: List of dictionaries (e.g., {"close": value})
#             if isinstance(price_values, dict):
#                 close_price = float(price_values["close"])
#             # Case 2: List of lists (e.g., [timestamp, open, high, low, close, volume])
#             elif isinstance(price_values, list):
#                 close_price = float(price_values[4])  # Close price is typically at index 4
#             else:
#                 raise ValueError("Unsupported data format")
            
#             # Check for NaN values
#             if math.isnan(close_price):
#                 continue
#             close_prices.append(close_price)
        
#         except (KeyError, IndexError, ValueError) as e:
#             print(f"Error processing price data: {str(e)}")
#             continue
    
#     return close_prices


# Calculate spread
def calculate_spread(series_1, series_2, hedge_ratio):
    spread = pd.Series(series_1) - (pd.Series(series_2) * hedge_ratio)
    return spread
 
# Calculate co-integration
def calculate_cointegration(series_1, series_2):
    coint_flag = 0
    coint_res = coint(series_1, series_2)
    coint_t = coint_res[0]
    p_value = coint_res[1]
    critical_value = coint_res[2][1]
    model = sm.OLS(series_1, series_2).fit()
    hedge_ratio = model.params[0]
    spread = calculate_spread(series_1, series_2, hedge_ratio)
    zero_crossings = len(np.where(np.diff(np.sign(spread)))[0])
    if p_value < 0.5 and coint_t < critical_value:
        coint_flag = 1
    return (coint_flag, round(p_value, 2), round(coint_t, 2), round(critical_value, 2), round(hedge_ratio, 2), zero_crossings)
 
 
# Calculate Z-Score
def calculate_zscore(spread):
    df = pd.DataFrame(spread)
    mean = df.rolling(center=False, window=z_score_window).mean()
    std = df.rolling(center=False, window=z_score_window).std()
    x = df.rolling(center=False, window=1).mean()
    df["ZSCORE"] = (x - mean) / std
    return df["ZSCORE"].astype(float).values
 
 
 
#Calculate Cointegrated Pairs
def get_cointegrated_pairs(prices):
 
    #loop through coins and check for cointegration
    cointegrated_pair_list = []
    included_list = [] #helps us avoid including the pairs twice
 
    for sym_1 in prices.keys():
 
 
        # Check each coin against the first (sym_1)
            for sym_2 in prices.keys():
                if sym_2 != sym_1:
 
                # Get unique combination id and ensure one off check
                    sorted_characters = sorted(sym_1 + sym_2)
 
                    unique = "".join(sorted_characters) 
                    if unique in included_list:
                        continue #alternatively you can use break
 
                    # Get close prices
                    series_1 = extract_close_prices(prices[sym_1])
                    series_2 = extract_close_prices(prices[sym_2])
 
 
                    # Check for cointegration and add cointegrated pair
                    coint_flag, p_value, t_value, c_value, hedge_ratio, zero_crossings = calculate_cointegration(series_1, series_2)
                    if coint_flag == 1:
                        included_list.append(unique)
                        cointegrated_pair_list.append({
                            "sym_1": sym_1,
                            "sym_2": sym_2,
                            "p_value": p_value,
                            "t_value": t_value,
                            "c_value": c_value,
                            "hedge_ratio": hedge_ratio,
                            "zero_crossings": zero_crossings
                        })
 
    # Output results
    df_coint = pd.DataFrame(cointegrated_pair_list)
    df_coint = df_coint.sort_values("zero_crossings", ascending=False)
    df_coint.to_csv("cointegrated_pairs_trial.csv")
    return df_coint



# Add new imports
from statsmodels.tsa.stattools import adfuller
from pykalman import KalmanFilter
import numpy as np

# def hurst_exponent(spread):
#     """Calculate Hurst exponent for mean reversion detection"""
#     lags = range(2, 100)
#     tau = [np.std(np.subtract(spread[lag:], spread[:-lag])) for lag in lags]
#     return np.polyfit(np.log(lags), np.log(tau), 1)[0] * 2

def hurst_exponent(spread):
    lags = range(2, 100)
    tau = [np.std(np.subtract(spread[lag:], spread[:-lag])) + 1e-8 for lag in lags]
    return np.polyfit(np.log(lags), np.log(tau), 1)[0] * 2

def kalman_filter(y, x):
    """Kalman filter for dynamic hedge ratio estimation"""
    try:
        # Convert to numpy arrays and ensure numeric types
        y_values = y.to_numpy(dtype=np.float64).reshape(-1, 1)
        x_values = x.to_numpy(dtype=np.float64).reshape(-1, 1, 1)
        
        # Properly format Kalman parameters as matrices
        kf = KalmanFilter(
            transition_matrices=np.eye(1),  # State transition matrix
            observation_matrices=x_values,  # Shape (n_timesteps, 1, 1)
            initial_state_mean=np.zeros(1),
            initial_state_covariance=np.eye(1),
            observation_covariance=np.eye(1),
            transition_covariance=np.eye(1)*0.01
        )
        
        # Filter using the actual price values
        state_means, _ = kf.filter(y_values)
        return state_means.flatten()
    
    except Exception as e:
        print(f"Kalman filter error: {str(e)}")
        return np.zeros(len(y))


def dynamic_ols(y, x, window=21):
    """Rolling OLS implementation"""
    hedge_ratios = []
    for i in range(window, len(x)):
        model = sm.OLS(y[i-window:i], x[i-window:i]).fit()
        hedge_ratios.append(model.params[0])
    return np.array(hedge_ratios)

# def enhanced_cointegration(series_1, series_2):
#     """Comprehensive cointegration check with multiple tests"""
#     # Basic Engle-Granger test
#     eg_test = coint(series_1, series_2)
    
#     # ADF test on spread
#     spread = series_1 - series_2
#     adf_result = adfuller(spread)
    
#     # Hurst exponent
#     hurst = hurst_exponent(spread)
    
#     # Kalman filter
#     kalman_hedge = kalman_filter(series_1, series_2)
#     kalman_spread = series_1 - kalman_hedge * series_2
#     kalman_adf = adfuller(kalman_spread)
    
#     # Dynamic OLS
#     dyn_hedge = dynamic_ols(series_1, series_2)
#     dyn_spread = series_1[-len(dyn_hedge):] - dyn_hedge * series_2[-len(dyn_hedge):]
#     dyn_adf = adfuller(dyn_spread)
    
#     # Composite score calculation
#     score = (eg_test[1] + adf_result[1] + kalman_adf[1] + dyn_adf[1])/4
    
#     return {
#         'eg_pvalue': eg_test[1],
#         'adf_pvalue': adf_result[1],
#         'hurst': hurst,
#         'kalman_pvalue': kalman_adf[1],
#         'dynols_pvalue': dyn_adf[1],
#         'composite_score': score
#     }

def calculate_half_life(spread):
    """Calculate the half-life of mean reversion for the spread"""
    spread_series = pd.Series(spread)
    delta_spread = spread_series.diff().dropna()
    lagged_spread = spread_series.shift(1).dropna()
    data = pd.DataFrame({"lagged_spread": lagged_spread, "delta_spread": delta_spread})
    data.dropna(inplace=True)
    model = sm.OLS(data["delta_spread"], data["lagged_spread"]).fit()
    lambda_ = model.params[0]
    half_life = -np.log(2) / lambda_ if lambda_ < 0 else np.inf
    return max(0, half_life)

def enhanced_cointegration(series_1, series_2):
    """Comprehensive cointegration check with multiple tests"""
    # Basic Engle-Granger test
    eg_test = coint(series_1, series_2)
    # ADF test on spread
    spread = series_1 - series_2
    adf_result = adfuller(spread)
    # Hurst exponent
    hurst = hurst_exponent(spread)
    # Half-life of mean reversion
    half_life = calculate_half_life(spread)
    # Kalman filter
    kalman_hedge = kalman_filter(series_1, series_2)
    kalman_spread = series_1 - kalman_hedge * series_2
    kalman_adf = adfuller(kalman_spread)
    # Dynamic OLS
    dyn_hedge = dynamic_ols(series_1, series_2)
    dyn_spread = series_1[-len(dyn_hedge):] - dyn_hedge * series_2[-len(dyn_hedge):]
    dyn_adf = adfuller(dyn_spread)
    # Composite score calculation
    score = (eg_test[1] + adf_result[1] + kalman_adf[1] + dyn_adf[1]) / 4
    return {
        'eg_pvalue': eg_test[1],
        'adf_pvalue': adf_result[1],
        'hurst': hurst,
        'kalman_pvalue': kalman_adf[1],
        'dynols_pvalue': dyn_adf[1],
        'composite_score': score,
        'half_life': half_life
    }