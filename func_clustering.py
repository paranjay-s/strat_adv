import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from kneed import KneeLocator
from func_pca_analysis import static_pca, rolling_pca


def calculate_metrics(prices):
    """Calculate returns and volatility"""
    returns = pd.DataFrame({sym: np.log(prices[sym]).diff().dropna() for sym in prices.keys()})
    volatility = returns.rolling(21).std() * np.sqrt(252)
    return returns.mean(), volatility.mean()

# from sklearn.preprocessing import MinMaxScaler

# def calculate_metrics(prices):
#     """
#     Calculate returns and volatility from close prices and scale them between 0 and 1.
#     """
#     # Convert prices to numeric and drop NA
#     returns = prices.pct_change().dropna()
#     log_returns = np.log(prices / prices.shift(1)).dropna()
#     volatility = log_returns.rolling(21).std() * np.sqrt(252)
    
#     # Scale returns and volatility between 0 and 1
#     scaler = MinMaxScaler()
#     scaled_returns = scaler.fit_transform(returns.mean().values.reshape(-1, 1)).flatten()
#     scaled_volatility = scaler.fit_transform(volatility.mean().values.reshape(-1, 1)).flatten()
    
#     return pd.Series(scaled_returns, index=returns.columns), pd.Series(scaled_volatility, index=volatility.columns)


def optimal_clusters(features):
    """Find optimal number of clusters using elbow method"""
    distortions = []
    K = range(1, 15)
    for k in K:
        kmeans = KMeans(n_clusters=k).fit(features)
        distortions.append(kmeans.inertia_)
    return KneeLocator(K, distortions, curve='convex', direction='decreasing').elbow

# def cluster_symbols(price_data):
#     """Cluster symbols using K-means"""
#     returns, volatility = calculate_metrics(price_data)
#     features = pd.DataFrame({'Returns': returns, 'Volatility': volatility}).dropna()
    
    # # Feature scaling
    # scaler = StandardScaler()
    # scaled_features = scaler.fit_transform(features)
    
    # # Cluster analysis
    # n_clusters = optimal_clusters(scaled_features)
    # kmeans = KMeans(n_clusters=n_clusters).fit(scaled_features)
    
#     return pd.Series(kmeans.labels_, index=features.index)


from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler
from kneed import KneeLocator

def cluster_symbols(features):
    """
    Cluster symbols using K-means with PCA-transformed features.
    """
    # Feature scaling
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)
    
    # # Find optimal number of clusters
    # distortions = []
    # K = range(1, 15)
    # for k in K:
    #     kmeans = KMeans(n_clusters=k).fit(scaled_features)
    #     distortions.append(kmeans.inertia_)
    # n_clusters = KneeLocator(K, distortions, curve='convex', direction='decreasing').elbow
    
    # # Perform clustering
    # kmeans = KMeans(n_clusters=n_clusters).fit(scaled_features)
    
    # Cluster analysis
    n_clusters = optimal_clusters(scaled_features)
    kmeans = KMeans(n_clusters=n_clusters).fit(scaled_features)

    return pd.Series(kmeans.labels_, index=features.index)


from sklearn.cluster import DBSCAN

def dbscan_clustering(features, eps=0.5, min_samples=5):
    """
    Cluster symbols using DBSCAN.
    """
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)
    dbscan = DBSCAN(eps=eps, min_samples=min_samples).fit(scaled_features)
    return pd.Series(dbscan.labels_, index=features.index)

# from sklearn.cluster import DBSCAN

# def dbscan_clustering(features, eps=0.5, min_samples=5):
#     """Cluster symbols using DBSCAN."""
#     scaler = StandardScaler()
#     scaled_features = scaler.fit_transform(features)
#     dbscan = DBSCAN(eps=eps, min_samples=min_samples).fit(scaled_features)
#     return pd.Series(dbscan.labels_, index=features.index)