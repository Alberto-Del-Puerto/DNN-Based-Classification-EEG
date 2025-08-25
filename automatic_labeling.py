import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA  
from scipy import stats
import pandas as pd


def automatic_labeling(psd_data, msc_data, entropy_data):
    # 1. Extraer características clave
    alpha_power = np.mean(psd_data, axis=1)  # Banda 
    mean_msc = np.mean(msc_data, axis=1)               # Coherencia promedio
    entropy = entropy_data.flatten()                     # Entropía
   
    print('alpha power',alpha_power)
    print('mean msc',mean_msc)
    print('entropy',entropy)
    
    mean_msc= pd.Series(mean_msc).ffill().values
    
    # 2. Normalizar características
    features = np.column_stack([
        stats.zscore(alpha_power),
        stats.zscore(mean_msc),
        stats.zscore(entropy)
    ])
    
    # 3. Clustering con KMeans (2 clusters: concentrado vs. no concentrado)
    kmeans = KMeans(n_clusters=2, random_state=42)
    clusters = kmeans.fit_predict(features)
    
    # 4. Asignar etiquetas basadas en fisiología:
    # - Cluster con MENOR potencia alfa + MAYOR MSC + MAYOR entropía → Concentrado (1)
    # - Cluster opuesto → No concentrado (0)
    cluster_means = np.array([
        [np.mean(alpha_power[clusters == 0]), np.mean(mean_msc[clusters == 0]), np.mean(entropy[clusters == 0])],
        [np.mean(alpha_power[clusters == 1]), np.mean(mean_msc[clusters == 1]), np.mean(entropy[clusters == 1])]
    ])
    
    # Identificar cuál cluster cumple los criterios de concentración
    concentration_cluster = np.argmin(cluster_means[:, 0] - cluster_means[:, 1] - cluster_means[:, 2])
    labels = np.where(clusters == concentration_cluster, 1, 0)
    
    return labels,alpha_power,mean_msc,entropy



def automatic_labeling_pca(psd_data, msc_data, entropy_data):
    """
    Assigns concentration labels using PCA instead of KMeans clustering
    
    Args:
        psd_data: ndarray of shape (n_samples, n_frequencies)
        msc_data: ndarray of shape (n_samples, n_frequencies) 
        entropy_data: ndarray of shape (n_samples, 1)
    
    Returns:
        tuple: (labels, alpha_power, mean_msc, entropy)
    """
    # 1. Extract key features
    alpha_power = np.mean(psd_data, axis=1)  # Mean power across all frequencies
    mean_msc = np.mean(msc_data, axis=1)     # Mean coherence
    entropy = entropy_data.flatten()         # Spectral entropy
    
    # Handle possible NaN values
    mean_msc = pd.Series(mean_msc).ffill().values
    
    # 2. Normalize features
    features = np.column_stack([
        stats.zscore(alpha_power),
        stats.zscore(mean_msc),
        stats.zscore(entropy)
    ])
    
       # 3. Apply PCA
    pca = PCA(n_components=1)
    principal_component = pca.fit_transform(features).flatten()
    
    # 4. Assign labels based on physiological interpretation:
    # We now assume that higher alpha power corresponds to concentration (higher PC1 values).
    # In this case, higher PC1 values will correspond to higher alpha power + higher MSC + higher entropy
    median_cutoff = np.median(principal_component)
    labels = np.where(principal_component > median_cutoff, 1, 0)
    
    # 5. Verify physiological consistency
    concentration_means = np.mean(features[labels == 1], axis=0)
    non_concentration_means = np.mean(features[labels == 0], axis=0)
    
    print("PCA Component Weights:", pca.components_[0])
    print("Concentration group means (alpha, MSC, entropy):", concentration_means)
    print("Non-concentration group means:", non_concentration_means)
   
    return labels, alpha_power, mean_msc, entropy