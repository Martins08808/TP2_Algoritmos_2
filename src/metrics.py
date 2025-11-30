import numpy as np
from sklearn.metrics import silhouette_score, adjusted_rand_score

def atribuir_labels_clusters(matriz_dist, indices_centros):
    """
    Dada a matriz de distâncias e os centros escolhidos,
    atribui cada ponto ao centro mais próximo (Labeling).
    
    Retorna:
        labels_pred: Array (N,) com o índice do cluster (0 a k-1) para cada ponto.
    """
    # Seleciona apenas as colunas correspondentes aos centros escolhidos
    # matriz_dist shape: (N, N) -> sub_matriz shape: (N, K)
    sub_matriz = matriz_dist[:, indices_centros]
    
    # Para cada linha (ponto), descobre qual coluna (centro) tem a menor distância
    labels_pred = np.argmin(sub_matriz, axis=1)
    
    return labels_pred

# def calcular_metricas_validacao(matriz_dist, labels_pred, labels_true):
#     """
#     Calcula Silhouette (usando a matriz de distâncias pré-calculada) e ARI.
#     """
#     resultados = {}
    
#     # [cite_start]1. Adjusted Rand Index (ARI) [cite: 38]
#     # Compara o agrupamento obtido com o gabarito (labels_true)
#     if labels_true is not None:
#         resultados['ARI'] = adjusted_rand_score(labels_true, labels_pred)
#     else:
#         resultados['ARI'] = np.nan # Caso não tenha ground truth

#     # [cite_start]2. Silhouette Coefficient [cite: 38]
#     # Mede a coesão e separação dos clusters.
#     # IMPORTANTE: Usamos metric='precomputed' para respeitar a métrica (Minkowski/Mahalanobis)
#     # que você já calculou na matriz_dist.
#     try:
#         # Silhouette exige pelo menos 2 clusters e menos que N-1
#         n_labels = len(set(labels_pred))
#         if 1 < n_labels < len(labels_pred):
#             resultados['Silhouette'] = silhouette_score(
#                 matriz_dist, 
#                 labels_pred, 
#                 metric='precomputed'
#             )
#         else:
#             resultados['Silhouette'] = -1.0 # Falha ou cluster único
#     except Exception as e:
#         print(f"Erro ao calcular Silhouette: {e}")
#         resultados['Silhouette'] = -1.0

#     return resultados