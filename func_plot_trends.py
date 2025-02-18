# from func_cointegration import extract_close_prices
# from func_cointegration import calculate_cointegration
# from func_cointegration import calculate_spread
# from func_cointegration import calculate_zscore
# import matplotlib.pyplot as plt
# import pandas as pd
 
 
# # Plot prices and trends
# def plot_trends(sym_1, sym_2, price_data):
 
#     # Extract prices
#     prices_1 = extract_close_prices(price_data[sym_1])
#     prices_2 = extract_close_prices(price_data[sym_2])
 
#     # Get spread and zscore
#     coint_flag, p_value, t_value, c_value, hedge_ratio, zero_crossing = calculate_cointegration(prices_1, prices_2)
#     spread = calculate_spread(prices_1, prices_2, hedge_ratio)
#     zscore = calculate_zscore(spread)
 
#     # Calculate percentage changes
#     df = pd.DataFrame(columns=[sym_1, sym_2])
#     df[sym_1] = prices_1
#     df[sym_2] = prices_2
#     df[f"{sym_1}_pct"] = df[sym_1] / prices_1[0]
#     df[f"{sym_2}_pct"] = df[sym_2] / prices_2[0]
#     series_1 = df[f"{sym_1}_pct"].astype(float).values
#     series_2 = df[f"{sym_2}_pct"].astype(float).values
 
#     # Save results for backtesting
#     df_2 = pd.DataFrame()
#     df_2[sym_1] = prices_1
#     df_2[sym_2] = prices_2
#     df_2["Spread"] = spread
#     df_2["ZScore"] = zscore
#     df_2.to_csv("trail_backtest.csv")
#     print("File for backtesting saved.")
 
#     # Plot charts
#     fig, axs = plt.subplots(3, figsize=(16, 8))
#     fig.suptitle(f"Price and Spread - {sym_1} vs {sym_2}")
#     axs[0].plot(series_1)
#     axs[0].plot(series_2)
#     axs[1].plot(spread)
#     axs[2].plot(zscore)
#     plt.show()


import warnings
warnings.simplefilter(action='ignore', category=FutureWarning) 
import numpy as np

from func_cointegration import extract_close_prices
from func_cointegration import calculate_cointegration
from func_cointegration import calculate_spread
from func_cointegration import calculate_zscore
from func_copulas import select_best_copula, validate_data  # Import the copula selection function
import matplotlib.pyplot as plt
import pandas as pd

from func_cointegration import extract_close_prices
from func_cointegration import calculate_cointegration
from func_cointegration import calculate_spread
from func_cointegration import calculate_zscore
from func_copulas import select_best_copula  # Import the copula selection function
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Plot prices, trends, and copula
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

from func_cointegration import extract_close_prices
from func_cointegration import calculate_cointegration
from func_cointegration import calculate_spread
from func_cointegration import calculate_zscore
from scipy.stats import norm, gaussian_kde
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# def plot_trends(sym_1, sym_2, price_data):
#     # Extract prices
#     prices_1 = extract_close_prices(price_data[sym_1])
#     prices_2 = extract_close_prices(price_data[sym_2])

#     # Get spread and zscore
#     coint_flag, p_value, t_value, c_value, hedge_ratio, zero_crossing = calculate_cointegration(prices_1, prices_2)
#     spread = calculate_spread(prices_1, prices_2, hedge_ratio)
#     zscore = calculate_zscore(spread)

#     # Calculate percentage changes
#     df = pd.DataFrame(columns=[sym_1, sym_2])
#     df[sym_1] = prices_1
#     df[sym_2] = prices_2
#     df[f"{sym_1}_pct"] = df[sym_1] / prices_1[0]
#     df[f"{sym_2}_pct"] = df[sym_2] / prices_2[0]
#     series_1 = df[f"{sym_1}_pct"].astype(float).values
#     series_2 = df[f"{sym_2}_pct"].astype(float).values

#     # Save results for backtesting
#     df_2 = pd.DataFrame()
#     df_2[sym_1] = prices_1
#     df_2[sym_2] = prices_2
#     df_2["Spread"] = spread
#     df_2["ZScore"] = zscore
#     df_2.to_csv("3_backtest_file.csv")
#     print("File for backtesting saved.")

#     # Calculate Bollinger Bands for Z-Score
#     window = 20  # Typical Bollinger Band window size
#     rolling_mean = pd.Series(zscore).rolling(window=window).mean()
#     rolling_std = pd.Series(zscore).rolling(window=window).std()
#     upper_band = rolling_mean + (2 * rolling_std)
#     lower_band = rolling_mean - (2 * rolling_std)

#     # Plot charts
#     fig, axs = plt.subplots(5, figsize=(16, 20))  # Increase overall figure height
#     fig.suptitle(f"Price, Spread, Z-Score, and Copula Analysis - {sym_1} vs {sym_2}")

#     # Price Chart
#     axs[0].plot(series_1, label=f"{sym_1} % Change")
#     axs[0].plot(series_2, label=f"{sym_2} % Change")
#     axs[0].set_title("Price Trends")
#     axs[0].legend()

#     # Spread Chart
#     axs[1].plot(spread, label="Spread", color="orange")
#     axs[1].set_title("Spread")
#     axs[1].legend()

#     # Z-Score Chart
#     axs[2].plot(zscore, label="Z-Score", color="green")
#     axs[2].set_title("Z-Score")
#     axs[2].legend()

#     # Z-Score with Bollinger Bands Chart
#     axs[3].plot(zscore, label="Z-Score", color="green")
#     axs[3].plot(rolling_mean, label="Middle Band", color="blue", linestyle="--")
#     axs[3].plot(upper_band, label="Upper Band", color="red", linestyle="--")
#     axs[3].plot(lower_band, label="Lower Band", color="red", linestyle="--")
#     axs[3].fill_between(range(len(zscore)), lower_band, upper_band, color="gray", alpha=0.2)
#     axs[3].set_title("Z-Score with Bollinger Bands")
#     axs[3].legend()

#     # Copula Chart
#     returns_1 = pd.Series(prices_1).pct_change().dropna()
#     returns_2 = pd.Series(prices_2).pct_change().dropna()

#     # Transform returns to quantiles (uniform marginals)
#     u = returns_1.rank(pct=True).values
#     v = returns_2.rank(pct=True).values

#     # Fit a Gaussian copula
#     from scipy.stats import multivariate_normal
#     mean = [0.5, 0.5]  # Center of the copula
#     cov = np.cov(u, v)  # Covariance matrix
#     copula = multivariate_normal(mean=mean, cov=cov)

#     # Generate grid for copula density
#     x = np.linspace(0.01, 0.99, 100)
#     y = np.linspace(0.01, 0.99, 100)
#     X, Y = np.meshgrid(x, y)
#     pos = np.dstack((X, Y))
#     Z = copula.pdf(pos)

#     # Define 95% boundary based on data quantiles
#     data_density = copula.pdf(np.column_stack((u, v)))
#     threshold = np.percentile(data_density, 5)  # 5th percentile (95% of data inside)

#     # Plot copula chart
#     axs[4].scatter(u, v, alpha=0.5, label="Ranked Returns")
#     axs[4].contour(X, Y, Z, levels=[threshold], colors="red", linestyles="--", label="95% Boundary")
#     axs[4].set_title(f"Copula Analysis with 95% Boundary")
#     axs[4].set_xlabel("Ranked Returns of Symbol 1")
#     axs[4].set_ylabel("Ranked Returns of Symbol 2")
#     axs[4].legend()

#     # Highlight outliers
#     outliers = []
#     for i in range(len(u)):
#         if copula.pdf([u[i], v[i]]) < threshold:
#             outliers.append((u[i], v[i]))
#     if outliers:
#         outlier_u, outlier_v = zip(*outliers)
#         axs[4].scatter(outlier_u, outlier_v, color="red", label="Outliers", marker="x")

#     # Adjust the size of the copula subplot
#     axs[4].set_aspect('equal', adjustable='box')  # Ensure equal aspect ratio
#     axs[4].set_xlim(0, 1)
#     axs[4].set_ylim(0, 1)

#     plt.tight_layout(rect=[0, 0, 1, 0.96])  # Adjust layout to fit the title
#     plt.show()



import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

from func_cointegration import extract_close_prices
from func_cointegration import calculate_cointegration
from func_cointegration import calculate_spread
from func_cointegration import calculate_zscore
from scipy.stats import norm, gaussian_kde
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec  # For custom subplot sizing
import pandas as pd
import numpy as np

# def plot_trends(sym_1, sym_2, price_data):
#     # Extract prices
#     prices_1 = extract_close_prices(price_data[sym_1])
#     prices_2 = extract_close_prices(price_data[sym_2])

#     # Get spread and zscore
#     coint_flag, p_value, t_value, c_value, hedge_ratio, zero_crossing = calculate_cointegration(prices_1, prices_2)
#     spread = calculate_spread(prices_1, prices_2, hedge_ratio)
#     zscore = calculate_zscore(spread)

#     # Calculate percentage changes
#     df = pd.DataFrame(columns=[sym_1, sym_2])
#     df[sym_1] = prices_1
#     df[sym_2] = prices_2
#     df[f"{sym_1}_pct"] = df[sym_1] / prices_1[0]
#     df[f"{sym_2}_pct"] = df[sym_2] / prices_2[0]
#     series_1 = df[f"{sym_1}_pct"].astype(float).values
#     series_2 = df[f"{sym_2}_pct"].astype(float).values

#     # Save results for backtesting
#     df_2 = pd.DataFrame()
#     df_2[sym_1] = prices_1
#     df_2[sym_2] = prices_2
#     df_2["Spread"] = spread
#     df_2["ZScore"] = zscore
#     df_2.to_csv("3_backtest_file.csv")
#     print("File for backtesting saved.")

#     # Calculate Bollinger Bands for Z-Score
#     window = 20  # Typical Bollinger Band window size
#     rolling_mean = pd.Series(zscore).rolling(window=window).mean()
#     rolling_std = pd.Series(zscore).rolling(window=window).std()
#     upper_band = rolling_mean + (2 * rolling_std)
#     lower_band = rolling_mean - (2 * rolling_std)

#     # Create a figure with custom grid specifications
#     fig = plt.figure(figsize=(20, 24))  # Increase overall figure size
#     gs = gridspec.GridSpec(6, 1, height_ratios=[1, 1, 1, 1, 2, 2])  # Allocate more height to the copula chart
#     fig.suptitle(f"Price, Spread, Z-Score, and Copula Analysis - {sym_1} vs {sym_2}", fontsize=16)

#     # Price Chart
#     ax1 = plt.subplot(gs[0])
#     ax1.plot(series_1, label=f"{sym_1} % Change")
#     ax1.plot(series_2, label=f"{sym_2} % Change")
#     ax1.set_title("Price Trends")
#     ax1.legend()

#     # Spread Chart
#     ax2 = plt.subplot(gs[1])
#     ax2.plot(spread, label="Spread", color="orange")
#     ax2.set_title("Spread")
#     ax2.legend()

#     # Z-Score Chart
#     ax3 = plt.subplot(gs[2])
#     ax3.plot(zscore, label="Z-Score", color="green")
#     ax3.set_title("Z-Score")
#     ax3.legend()

#     # Z-Score with Bollinger Bands Chart
#     ax4 = plt.subplot(gs[3])
#     ax4.plot(zscore, label="Z-Score", color="green")
#     ax4.plot(rolling_mean, label="Middle Band", color="blue", linestyle="--")
#     ax4.plot(upper_band, label="Upper Band", color="red", linestyle="--")
#     ax4.plot(lower_band, label="Lower Band", color="red", linestyle="--")
#     ax4.fill_between(range(len(zscore)), lower_band, upper_band, color="gray", alpha=0.2)
#     ax4.set_title("Z-Score with Bollinger Bands")
#     ax4.legend()

#     # Copula Chart
#     returns_1 = pd.Series(prices_1).pct_change().dropna()
#     returns_2 = pd.Series(prices_2).pct_change().dropna()

#     # Transform returns to quantiles (uniform marginals)
#     u = returns_1.rank(pct=True).values
#     v = returns_2.rank(pct=True).values

#     # Fit a Gaussian copula
#     from scipy.stats import multivariate_normal
#     mean = [0.5, 0.5]  # Center of the copula
#     cov = np.cov(u, v)  # Covariance matrix
#     copula = multivariate_normal(mean=mean, cov=cov)

#     # Generate grid for copula density
#     x = np.linspace(0.01, 0.99, 100)
#     y = np.linspace(0.01, 0.99, 100)
#     X, Y = np.meshgrid(x, y)
#     pos = np.dstack((X, Y))
#     Z = copula.pdf(pos)

#     # Define 95% boundary based on data quantiles
#     data_density = copula.pdf(np.column_stack((u, v)))
#     threshold = np.percentile(data_density, 5)  # 5th percentile (95% of data inside)

#     # Plot copula chart
#     ax5 = plt.subplot(gs[4:6])  # Use two rows for the copula chart
#     ax5.scatter(u, v, alpha=0.5, label="Ranked Returns")
#     ax5.contour(X, Y, Z, levels=[threshold], colors="red", linestyles="--", label="95% Boundary")
#     ax5.set_title(f"Copula Analysis with 95% Boundary", fontsize=14)
#     ax5.set_xlabel("Ranked Returns of Symbol 1")
#     ax5.set_ylabel("Ranked Returns of Symbol 2")
#     ax5.legend()

#     # Highlight outliers
#     outliers = []
#     for i in range(len(u)):
#         if copula.pdf([u[i], v[i]]) < threshold:
#             outliers.append((u[i], v[i]))
#     if outliers:
#         outlier_u, outlier_v = zip(*outliers)
#         ax5.scatter(outlier_u, outlier_v, color="red", label="Outliers", marker="x")

#     # Adjust the size and aspect ratio of the copula subplot
#     ax5.set_aspect('equal', adjustable='box')  # Ensure equal aspect ratio
#     ax5.set_xlim(0, 1)
#     ax5.set_ylim(0, 1)

#     plt.tight_layout(rect=[0, 0, 1, 0.96])  # Adjust layout to fit the title
#     plt.show()




import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

from func_cointegration import extract_close_prices
from func_cointegration import calculate_cointegration
from func_cointegration import calculate_spread
from func_cointegration import calculate_zscore
from scipy.stats import norm, gaussian_kde
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def plot_trends(sym_1, sym_2, price_data):
    # Extract prices
    prices_1 = extract_close_prices(price_data[sym_1])
    prices_2 = extract_close_prices(price_data[sym_2])

    # Get spread and zscore
    coint_flag, p_value, t_value, c_value, hedge_ratio, zero_crossing = calculate_cointegration(prices_1, prices_2)
    spread = calculate_spread(prices_1, prices_2, hedge_ratio)
    zscore = calculate_zscore(spread)

    # Calculate percentage changes
    df = pd.DataFrame(columns=[sym_1, sym_2])
    df[sym_1] = prices_1
    df[sym_2] = prices_2
    df[f"{sym_1}_pct"] = df[sym_1] / prices_1[0]
    df[f"{sym_2}_pct"] = df[sym_2] / prices_2[0]
    series_1 = df[f"{sym_1}_pct"].astype(float).values
    series_2 = df[f"{sym_2}_pct"].astype(float).values

    # Save results for backtesting
    df_2 = pd.DataFrame()
    df_2[sym_1] = prices_1
    df_2[sym_2] = prices_2
    df_2["Spread"] = spread
    df_2["ZScore"] = zscore
    df_2.to_csv("3_backtest_file.csv")
    print("File for backtesting saved.")

    # Calculate Bollinger Bands for Z-Score
    window = 20  # Typical Bollinger Band window size
    rolling_mean = pd.Series(zscore).rolling(window=window).mean()
    rolling_std = pd.Series(zscore).rolling(window=window).std()
    upper_band = rolling_mean + (2 * rolling_std)
    lower_band = rolling_mean - (2 * rolling_std)

    # Create a figure with custom grid specifications
    fig = plt.figure(figsize=(20, 24))  # Increase overall figure size
    gs = gridspec.GridSpec(6, 1, height_ratios=[1, 1, 1, 1, 2, 2])  # Allocate more height to the copula chart
    fig.suptitle(f"Price, Spread, Z-Score, and Copula Analysis - {sym_1} vs {sym_2}", fontsize=16)

    # Price Chart
    ax1 = plt.subplot(gs[0])
    ax1.plot(series_1, label=f"{sym_1} % Change")
    ax1.plot(series_2, label=f"{sym_2} % Change")
    ax1.set_title("Price Trends")
    ax1.legend()

    # Spread Chart
    ax2 = plt.subplot(gs[1])
    ax2.plot(spread, label="Spread", color="orange")
    ax2.set_title("Spread")
    ax2.legend()

    # Z-Score Chart
    ax3 = plt.subplot(gs[2])
    ax3.plot(zscore, label="Z-Score", color="green")
    ax3.set_title("Z-Score")
    ax3.legend()

    # Z-Score with Bollinger Bands Chart
    ax4 = plt.subplot(gs[3])
    ax4.plot(zscore, label="Z-Score", color="green")
    ax4.plot(rolling_mean, label="Middle Band", color="blue", linestyle="--")
    ax4.plot(upper_band, label="Upper Band", color="red", linestyle="--")
    ax4.plot(lower_band, label="Lower Band", color="red", linestyle="--")
    ax4.fill_between(range(len(zscore)), lower_band, upper_band, color="gray", alpha=0.2)
    ax4.set_title("Z-Score with Bollinger Bands")
    ax4.legend()

    # Copula Chart
    returns_1 = pd.Series(prices_1).pct_change().dropna()
    returns_2 = pd.Series(prices_2).pct_change().dropna()

    # Transform returns to quantiles (uniform marginals)
    u = returns_1.rank(pct=True).values
    v = returns_2.rank(pct=True).values

    # Fit a Gaussian copula
    from scipy.stats import multivariate_normal
    mean = [0.5, 0.5]  # Center of the copula
    cov = np.cov(u, v)  # Covariance matrix
    copula = multivariate_normal(mean=mean, cov=cov)

    # Generate grid for copula density
    x = np.linspace(0.01, 0.99, 100)
    y = np.linspace(0.01, 0.99, 100)
    X, Y = np.meshgrid(x, y)
    pos = np.dstack((X, Y))
    Z = copula.pdf(pos)

    # Define 95% boundary based on data quantiles
    data_density = copula.pdf(np.column_stack((u, v)))
    threshold = np.percentile(data_density, 5)  # 5th percentile (95% of data inside)

    # Plot copula chart
    ax5 = plt.subplot(gs[4:6])  # Use two rows for the copula chart
    ax5.scatter(u, v, alpha=0.5, label="Ranked Returns")
    ax5.contour(X, Y, Z, levels=[threshold], colors="red", linestyles="--", label="95% Boundary")
    ax5.set_title(f"Copula Analysis with 95% Boundary", fontsize=14)
    ax5.set_xlabel("Ranked Returns of Symbol 1")
    ax5.set_ylabel("Ranked Returns of Symbol 2")
    ax5.legend()

    # Highlight the latest data point
    latest_u = u[-1]  # Latest ranked return for symbol 1
    latest_v = v[-1]  # Latest ranked return for symbol 2
    if copula.pdf([latest_u, latest_v]) < threshold:
        ax5.scatter(latest_u, latest_v, color="red", label="Latest Data Point (Outlier)", marker="o", s=100, edgecolor="black")
        print("Latest data point is OUTSIDE the copula boundary. Potential trading opportunity!")
    else:
        ax5.scatter(latest_u, latest_v, color="green", label="Latest Data Point (Inside)", marker="o", s=100, edgecolor="black")
        print("Latest data point is INSIDE the copula boundary. No trading opportunity at present.")

    # Adjust the size and aspect ratio of the copula subplot
    ax5.set_aspect('equal', adjustable='box')  # Ensure equal aspect ratio
    ax5.set_xlim(0, 1)
    ax5.set_ylim(0, 1)

    plt.tight_layout(rect=[0, 0, 1, 0.96])  # Adjust layout to fit the title
    plt.show()