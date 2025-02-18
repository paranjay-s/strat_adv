import numpy as np
from sklearn.decomposition import PCA

def rolling_pca(series, window=60, n_components=3):
    """Perform rolling PCA analysis"""
    results = []
    for i in range(len(series) - window):
        window_data = series.iloc[i:i+window]
        pca = PCA(n_components=n_components)
        pca.fit(window_data)
        results.append(pca.components_)
    return np.array(results)

def static_pca(series, n_components=3):
    """Perform static PCA analysis"""
    pca = PCA(n_components=n_components)
    pca.fit(series)
    return pca.components_
