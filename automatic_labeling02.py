import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA  
from scipy import stats
import pandas as pd

"""
    Asigna etiquetas de concentración usando KMeans basado en el siguiente perfil fisiológico:
    
    Perfil Fisiológico para Banda Gamma:
    -----------------------------------
    | Característica       | Concentrado (1) | No Concentrado (0) |
    |----------------------|------------------|---------------------|
    | Potencia Gamma       | Alta (↑)         | Baja                |
    | Coherencia (MSC)     | Alta (↑)         | Baja                |
    | Entropía Espectral   | Baja (↓)         | Alta                |
    -----------------------------------
    
    Args:
        psd_data: Potencia espectral en banda gamma (n_samples × n_frequencies)
        msc_data: Coherencia espectral en banda gamma (n_samples × n_frequencies) 
        entropy_data: Entropía espectral (n_samples × 1)
    
    Returns:
        tuple: (labels, gamma_power, mean_msc, entropy)
"""

def automatic_labeling(psd_data, msc_data, entropy_data):
    """
    Asigna etiquetas de concentración usando KMeans basado en el perfil fisiológico de gamma.
    Versión corregida para consistencia con PCA.
    
    Perfil Fisiológico:
    - Concentrado (1): Alta gamma, Alta MSC, Baja entropía
    - No Concentrado (0): Baja gamma, Baja MSC, Alta entropía
    """
    # 1. Extraer características
    gamma_power = np.mean(psd_data, axis=1)
    mean_msc = np.mean(msc_data, axis=1)
    entropy = entropy_data.flatten()
    mean_msc = pd.Series(mean_msc).ffill().values
    
    # 2. Normalizar (invertir entropía para alinear direcciones)
    features = np.column_stack([
        stats.zscore(gamma_power),
        stats.zscore(mean_msc),
        stats.zscore(-entropy)  # Invertimos entropía
    ])
    
    # 3. Clustering
    kmeans = KMeans(n_clusters=2, random_state=42)
    clusters = kmeans.fit_predict(features)
    
    # 4. Asignación CONSISTENTE de etiquetas
    centroids = kmeans.cluster_centers_
    
    # Criterio unificado: gamma↑ + MSC↑ + (entropía↓)
    concentration_scores = centroids[:,0] + centroids[:,1] + centroids[:,2]
    concentration_cluster = np.argmax(concentration_scores)
    labels = np.where(clusters == concentration_cluster, 1, 0)
    
    # Validación final de coherencia
    if np.mean(gamma_power[labels==1]) < np.mean(gamma_power[labels==0]):
        labels = 1 - labels  # Invertir si no cumple perfil
    
    print("\nValidación KMeans:")
    print(f"Gamma: 1={np.mean(gamma_power[labels==1]):.2f} vs 0={np.mean(gamma_power[labels==0]):.2f}")
    print(f"MSC: 1={np.mean(mean_msc[labels==1]):.2f} vs 0={np.mean(mean_msc[labels==0]):.2f}")
    print(f"Entropía: 1={np.mean(entropy[labels==1]):.2f} vs 0={np.mean(entropy[labels==0]):.2f}")
    
    return labels, gamma_power, mean_msc, entropy

def automatic_labeling_pca(psd_data, msc_data, entropy_data):
    """
    Asigna etiquetas de concentración usando PCA, alineado con el mismo perfil que KMeans.
    
    Perfil Fisiológico:
    - Concentrado (1): Alta gamma, Alta MSC, Baja entropía
    - No Concentrado (0): Baja gamma, Baja MSC, Alta entropía
    """
    # 1. Extraer características (igual que KMeans)
    gamma_power = np.mean(psd_data, axis=1)
    mean_msc = np.mean(msc_data, axis=1)
    entropy = entropy_data.flatten()
    mean_msc = pd.Series(mean_msc).ffill().values
    
    # 2. Normalizar (mismo tratamiento que KMeans)
    features = np.column_stack([
        stats.zscore(gamma_power),
        stats.zscore(mean_msc),
        stats.zscore(-entropy)  # Invertimos entropía
    ])
    
    # 3. PCA
    pca = PCA(n_components=1)
    principal_component = pca.fit_transform(features).flatten()
    
    # 4. Asegurar dirección correcta del componente
    if pca.components_[0, 0] + pca.components_[0, 1] - pca.components_[0, 2] < 0:
        principal_component = -principal_component
    
    # Asignar etiquetas (mayor PC = concentrado)
    labels = np.where(principal_component > np.median(principal_component), 1, 0)
    
    # Validación final idéntica a KMeans
    if np.mean(gamma_power[labels==1]) < np.mean(gamma_power[labels==0]):
        labels = 1 - labels
    
    print("\nValidación PCA:")
    print(f"Gamma: 1={np.mean(gamma_power[labels==1]):.2f} vs 0={np.mean(gamma_power[labels==0]):.2f}")
    print(f"MSC: 1={np.mean(mean_msc[labels==1]):.2f} vs 0={np.mean(mean_msc[labels==0]):.2f}")
    print(f"Entropía: 1={np.mean(entropy[labels==1]):.2f} vs 0={np.mean(entropy[labels==0]):.2f}")
    
    return labels, gamma_power, mean_msc, entropy

# Función para verificar consistencia entre métodos
def check_consistency(psd_data, msc_data, entropy_data):
    labels_k, gamma, msc, ent = automatic_labeling(psd_data, msc_data, entropy_data)
    labels_p, _, _, _ = automatic_labeling_pca(psd_data, msc_data, entropy_data)
    
    concordance = np.mean(labels_k == labels_p)
    print(f"\nConcordancia entre KMeans y PCA: {concordance*100:.1f}%")
    
    if concordance < 0.9:
        print("Advertencia: Los métodos difieren significativamente")
        print("Revisar distribución de características:")
        print(pd.DataFrame({
            'Gamma': gamma,
            'MSC': msc,
            'Entropía': ent
        }).describe())
    
    return labels_k, labels_p
  