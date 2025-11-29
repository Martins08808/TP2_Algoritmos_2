import numpy as np
import pandas as pd  # Adicionado para manipulação de CSV
import os
from sklearn import datasets
from sklearn.preprocessing import StandardScaler

# ==========================================
# CONFIGURAÇÕES GLOBAIS
# ==========================================
N_SAMPLES = 750
OUTPUT_DIR = "synthetic"  # Alterado nome da pasta para refletir o formato
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
    # Gera nomes de colunas dinamicamente (feature_1, feature_2)
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
        
        # -------------------------------------------------
        # TIPO 1: Noisy Circles
        # -------------------------------------------------
        factor = 0.3 + (i * 0.08)
        noise = 0.04 + (i * 0.01)
        X, y = datasets.make_circles(n_samples=N_SAMPLES, factor=factor, noise=noise, random_state=seed_iter)
        salvar_dataset(X, y, f"sklearn_01_circles_v{i}.csv")

        # -------------------------------------------------
        # TIPO 2: Noisy Moons
        # -------------------------------------------------
        noise_moons = 0.05 + (i * 0.015)
        X, y = datasets.make_moons(n_samples=N_SAMPLES, noise=noise_moons, random_state=seed_iter)
        salvar_dataset(X, y, f"sklearn_02_moons_v{i}.csv")

        # -------------------------------------------------
        # TIPO 3: Blobs
        # -------------------------------------------------
        n_centers = 3 + (i % 3) 
        cluster_std = 1.0 + (i * 0.1)
        X, y = datasets.make_blobs(n_samples=N_SAMPLES, centers=n_centers, cluster_std=cluster_std, random_state=seed_iter)
        salvar_dataset(X, y, f"sklearn_03_blobs_v{i}.csv")

        # -------------------------------------------------
        # TIPO 4: No Structure
        # -------------------------------------------------
        rng = np.random.RandomState(seed_iter)
        X_rand = rng.rand(N_SAMPLES, 2)
        y_rand = np.zeros(N_SAMPLES)
        salvar_dataset(X_rand, y_rand, f"sklearn_04_no_structure_v{i}.csv")

        # -------------------------------------------------
        # TIPO 5: Anisotropic
        # -------------------------------------------------
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

        # -------------------------------------------------
        # TIPO 6: Varied Variances
        # -------------------------------------------------
        std_devs = [1.0, 2.5 - (0.2 * i), 0.5 + (0.05 * i)]
        X_varied, y_varied = datasets.make_blobs(n_samples=N_SAMPLES, centers=3, cluster_std=std_devs, random_state=seed_iter)
        salvar_dataset(X_varied, y_varied, f"sklearn_06_varied_v{i}.csv")

if __name__ == "__main__":
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        
    gerar_sklearn_toy_data()
    
    total_files = len([name for name in os.listdir(OUTPUT_DIR) if name.endswith(".csv")])
    print(f"\nConcluído! Total de arquivos CSV gerados em '{OUTPUT_DIR}': {total_files}")