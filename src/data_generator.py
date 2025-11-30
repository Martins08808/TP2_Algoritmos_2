import numpy as np
import pandas as pd
import os
from sklearn import datasets
from sklearn.preprocessing import StandardScaler

# ==========================================
# CONFIGURAÇÕES GLOBAIS
# ==========================================
N_SAMPLES = 750
OUTPUT_DIR = "synthetic"
SEED_BASE = 42

def salvar_dataset(X, y, nome_arquivo):
    """
    Salva o dataset em formato .csv (Comma Separated Values).
    Aplica StandardScaler antes de salvar.
    Estrutura: feature_1, feature_2, ..., label
    """
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        
    caminho_completo = os.path.join(OUTPUT_DIR, nome_arquivo)
    
    # Normalização (Média 0, Desvio Padrão 1)
    X_scaled = StandardScaler().fit_transform(X)
    
    # Criação do DataFrame para facilitar a exportação CSV
    colunas = [f"feature_{i+1}" for i in range(X_scaled.shape[1])]
    df = pd.DataFrame(X_scaled, columns=colunas)
    
    # Adiciona a coluna de target/label
    df['label'] = y
    
    # Salva em CSV sem o índice do pandas
    df.to_csv(caminho_completo, index=False)
    print(f"[OK] Gerado: {nome_arquivo} | Shape: {df.shape}")

# ==========================================
# 1ª ABORDAGEM: DATASETS TIPO SCIKIT-LEARN
# Total: 30 datasets (6 tipos x 5 variações)
# ==========================================
def gerar_sklearn_toy_data():
    print(f"\n--- Gerando 30 Datasets da Abordagem Scikit-Learn (CSV) ---")
    
    for i in range(1, 6):
        seed_iter = SEED_BASE + i
        
        # TIPO 1: Noisy Circles
        factor = 0.3 + (i * 0.08)
        noise = 0.04 + (i * 0.01)
        X, y = datasets.make_circles(n_samples=N_SAMPLES, factor=factor, noise=noise, random_state=seed_iter)
        salvar_dataset(X, y, f"sklearn_01_circles_v{i}.csv")

        # TIPO 2: Noisy Moons
        noise_moons = 0.05 + (i * 0.015)
        X, y = datasets.make_moons(n_samples=N_SAMPLES, noise=noise_moons, random_state=seed_iter)
        salvar_dataset(X, y, f"sklearn_02_moons_v{i}.csv")

        # TIPO 3: Blobs
        n_centers = 3 + (i % 3) 
        cluster_std = 1.0 + (i * 0.1)
        X, y = datasets.make_blobs(n_samples=N_SAMPLES, centers=n_centers, cluster_std=cluster_std, random_state=seed_iter)
        salvar_dataset(X, y, f"sklearn_03_blobs_v{i}.csv")

        # TIPO 4: No Structure
        rng = np.random.RandomState(seed_iter)
        X_rand = rng.rand(N_SAMPLES, 2)
        y_rand = np.zeros(N_SAMPLES)
        salvar_dataset(X_rand, y_rand, f"sklearn_04_no_structure_v{i}.csv")

        # TIPO 5: Anisotropic
        X_base, y_base = datasets.make_blobs(n_samples=N_SAMPLES, centers=3, random_state=seed_iter)
        
        theta = np.radians(15 * i + 10)
        stretch_x = 0.6 + (0.05 * i)
        stretch_y = 0.4 - (0.02 * i)
        
        rotation_matrix = np.array([[np.cos(theta), -np.sin(theta)], 
                                    [np.sin(theta), np.cos(theta)]])
        scale_matrix = np.array([[stretch_x, 0], 
                                 [0, stretch_y]])
        
        transformation = rotation_matrix @ scale_matrix
        X_aniso = np.dot(X_base, transformation)
        
        salvar_dataset(X_aniso, y_base, f"sklearn_05_aniso_v{i}.csv")

        # TIPO 6: Varied Variances
        std_devs = [1.0, 2.5 - (0.2 * i), 0.5 + (0.05 * i)]
        X_varied, y_varied = datasets.make_blobs(n_samples=N_SAMPLES, centers=3, cluster_std=std_devs, random_state=seed_iter)
        salvar_dataset(X_varied, y_varied, f"sklearn_06_varied_v{i}.csv")

# ==========================================
# 2ª ABORDAGEM: DISTRIBUIÇÃO NORMAL MULTIVARIADA
# Total: 20 datasets (10 configurações x 2 variações de forma)
# ==========================================

def criar_matriz_covariancia(sigma_x, sigma_y, theta=0, correlacao=0):
    """
    Cria uma matriz de covariância 2D para controlar forma do cluster.
    
    Parâmetros:
    - sigma_x: desvio padrão no eixo x
    - sigma_y: desvio padrão no eixo y
    - theta: ângulo de rotação em radianos
    - correlacao: correlação entre x e y (-1 a 1)
    
    Retorna:
    - Matriz de covariância 2x2
    """
    # Matriz de covariância base (sem rotação)
    cov = np.array([
        [sigma_x**2, correlacao * sigma_x * sigma_y],
        [correlacao * sigma_x * sigma_y, sigma_y**2]
    ])
    
    # Matriz de rotação
    cos_t = np.cos(theta)
    sin_t = np.sin(theta)
    R = np.array([
        [cos_t, -sin_t],
        [sin_t, cos_t]
    ])
    
    # Aplicar rotação: R * Cov * R^T
    cov_rotacionada = R @ cov @ R.T
    
    return cov_rotacionada

def gerar_cluster_multivariado(mean, cov, n_samples, random_state):
    """
    Gera um cluster usando distribuição normal multivariada.
    
    Parâmetros:
    - mean: vetor de média [x, y]
    - cov: matriz de covariância 2x2
    - n_samples: número de amostras
    - random_state: semente aleatória
    
    Retorna:
    - Array de pontos (n_samples, 2)
    """
    rng = np.random.RandomState(random_state)
    return rng.multivariate_normal(mean, cov, n_samples)

def gerar_multivariate_normal_data():
    """
    Gera 20 datasets usando distribuição normal multivariada com diferentes configurações:
    - Variando número de clusters (2 a 5)
    - Variando sobreposição (sem sobreposição até alta sobreposição)
    - Variando forma (circular vs elíptico)
    """
    print(f"\n--- Gerando 20 Datasets com Distribuição Normal Multivariada (CSV) ---")
    
    configuracoes = [
        # CONFIG 1: 3 clusters circulares, SEM sobreposição
        {
            'n_clusters': 3,
            'centers': [[0, 0], [10, 0], [5, 8]],
            'sigmas': [(1.0, 1.0), (1.0, 1.0), (1.0, 1.0)],  # circular: sigma_x ≈ sigma_y
            'rotations': [0, 0, 0],
            'correlacoes': [0, 0, 0],
            'descricao': 'circular_sem_overlap'
        },
        
        # CONFIG 2: 3 clusters elípticos, SEM sobreposição
        {
            'n_clusters': 3,
            'centers': [[0, 0], [10, 0], [5, 8]],
            'sigmas': [(2.0, 0.5), (2.0, 0.5), (2.0, 0.5)],  # elíptico: sigma_x >> sigma_y
            'rotations': [0, np.pi/4, np.pi/2],  # rotações diferentes
            'correlacoes': [0, 0, 0],
            'descricao': 'eliptico_sem_overlap'
        },
        
        # CONFIG 3: 4 clusters circulares, BAIXA sobreposição
        {
            'n_clusters': 4,
            'centers': [[0, 0], [6, 0], [0, 6], [6, 6]],
            'sigmas': [(1.5, 1.5), (1.5, 1.5), (1.5, 1.5), (1.5, 1.5)],
            'rotations': [0, 0, 0, 0],
            'correlacoes': [0, 0, 0, 0],
            'descricao': 'circular_baixa_overlap'
        },
        
        # CONFIG 4: 4 clusters elípticos, BAIXA sobreposição
        {
            'n_clusters': 4,
            'centers': [[0, 0], [6, 0], [0, 6], [6, 6]],
            'sigmas': [(2.5, 0.8), (2.5, 0.8), (2.5, 0.8), (2.5, 0.8)],
            'rotations': [0, np.pi/3, np.pi/2, -np.pi/4],
            'correlacoes': [0, 0, 0, 0],
            'descricao': 'eliptico_baixa_overlap'
        },
        
        # CONFIG 5: 3 clusters circulares, MÉDIA sobreposição
        {
            'n_clusters': 3,
            'centers': [[0, 0], [4, 0], [2, 3.5]],
            'sigmas': [(1.8, 1.8), (1.8, 1.8), (1.8, 1.8)],
            'rotations': [0, 0, 0],
            'correlacoes': [0, 0, 0],
            'descricao': 'circular_media_overlap'
        },
        
        # CONFIG 6: 3 clusters elípticos, MÉDIA sobreposição
        {
            'n_clusters': 3,
            'centers': [[0, 0], [4, 0], [2, 3.5]],
            'sigmas': [(3.0, 1.0), (3.0, 1.0), (3.0, 1.0)],
            'rotations': [np.pi/6, -np.pi/6, np.pi/3],
            'correlacoes': [0, 0, 0],
            'descricao': 'eliptico_media_overlap'
        },
        
        # CONFIG 7: 5 clusters circulares, ALTA sobreposição
        {
            'n_clusters': 5,
            'centers': [[0, 0], [2, 0], [4, 0], [1, 2], [3, 2]],
            'sigmas': [(1.5, 1.5), (1.5, 1.5), (1.5, 1.5), (1.5, 1.5), (1.5, 1.5)],
            'rotations': [0, 0, 0, 0, 0],
            'correlacoes': [0, 0, 0, 0, 0],
            'descricao': 'circular_alta_overlap'
        },
        
        # CONFIG 8: 5 clusters elípticos, ALTA sobreposição
        {
            'n_clusters': 5,
            'centers': [[0, 0], [2, 0], [4, 0], [1, 2], [3, 2]],
            'sigmas': [(2.5, 1.2), (2.5, 1.2), (2.5, 1.2), (2.5, 1.2), (2.5, 1.2)],
            'rotations': [0, np.pi/4, np.pi/2, -np.pi/4, np.pi/6],
            'correlacoes': [0, 0, 0, 0, 0],
            'descricao': 'eliptico_alta_overlap'
        },
        
        # CONFIG 9: 2 clusters circulares com correlação positiva
        {
            'n_clusters': 2,
            'centers': [[0, 0], [8, 8]],
            'sigmas': [(2.0, 2.0), (2.0, 2.0)],
            'rotations': [0, 0],
            'correlacoes': [0.7, 0.7],  # correlação positiva
            'descricao': 'circular_correlacao_positiva'
        },
        
        # CONFIG 10: 3 clusters elípticos com tamanhos diferentes
        {
            'n_clusters': 3,
            'centers': [[0, 0], [7, 0], [3.5, 6]],
            'sigmas': [(1.0, 0.3), (2.5, 0.8), (4.0, 1.5)],  # tamanhos crescentes
            'rotations': [0, np.pi/4, np.pi/3],
            'correlacoes': [0, 0, 0],
            'descricao': 'eliptico_tamanhos_diferentes'
        }
    ]
    
    dataset_id = 1
    
    for config in configuracoes:
        seed = SEED_BASE + dataset_id
        
        n_clusters = config['n_clusters']
        n_per_cluster = N_SAMPLES // n_clusters
        
        X_list = []
        y_list = []
        
        for cluster_id in range(n_clusters):
            mean = config['centers'][cluster_id]
            sigma_x, sigma_y = config['sigmas'][cluster_id]
            theta = config['rotations'][cluster_id]
            corr = config['correlacoes'][cluster_id]
            
            # Criar matriz de covariância
            cov = criar_matriz_covariancia(sigma_x, sigma_y, theta, corr)
            
            # Gerar pontos do cluster
            X_cluster = gerar_cluster_multivariado(
                mean=mean,
                cov=cov,
                n_samples=n_per_cluster,
                random_state=seed + cluster_id
            )
            
            X_list.append(X_cluster)
            y_list.append(np.full(n_per_cluster, cluster_id))
        
        # Concatenar todos os clusters
        X = np.vstack(X_list)
        y = np.concatenate(y_list)
        
        # Embaralhar os dados
        rng = np.random.RandomState(seed)
        indices = rng.permutation(len(X))
        X = X[indices]
        y = y[indices]
        
        # Salvar dataset
        nome_arquivo = f"multivar_{dataset_id:02d}_{config['descricao']}.csv"
        salvar_dataset(X, y, nome_arquivo)
        
        dataset_id += 1

# ==========================================
# FUNÇÃO PRINCIPAL
# ==========================================
if __name__ == "__main__":
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
    
    # Gerar datasets da 1ª abordagem (30 datasets)
    gerar_sklearn_toy_data()
    
    # Gerar datasets da 2ª abordagem (20 datasets)
    gerar_multivariate_normal_data()
    
    # Resumo final
    total_files = len([name for name in os.listdir(OUTPUT_DIR) if name.endswith(".csv")])
    print(f"\n{'='*60}")
    print(f"✓ GERAÇÃO CONCLUÍDA!")
    print(f"{'='*60}")
    print(f"Total de arquivos CSV gerados: {total_files}")
    print(f"Diretório: '{OUTPUT_DIR}/'")
    print(f"\nDistribuição:")
    print(f"  • Abordagem Scikit-Learn: 30 datasets")
    print(f"  • Abordagem Multivariada:  20 datasets")
    print(f"{'='*60}\n")