import numpy as np
import os
from sklearn import datasets
from sklearn.preprocessing import StandardScaler

# Configurações Globais
N_SAMPLES = 750  # Mínimo exigido é 700
OUTPUT_DIR = "data/synthetic"
SEED_BASE = 42

def gerar_dataset_teste(nome_arquivo="integration_test.npz"):
    """
    Gera um único dataset simples (Blobs) especificamente para testes de integração.
    Retorna o caminho do arquivo gerado.
    """
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        
    caminho_completo = os.path.join(OUTPUT_DIR, nome_arquivo)
    
    # Se já existe, não precisa recriar (economiza tempo no teste)
    if os.path.exists(caminho_completo):
        print(f"[TESTE] Dataset já existente encontrado: {caminho_completo}")
        return caminho_completo

    print(f"[TESTE] Gerando dataset sintético para teste...")
    # Gera 3 blobs simples para validar K=3
    X, y = datasets.make_blobs(n_samples=N_SAMPLES, centers=3, random_state=SEED_BASE)
    
    # Aplica a mesma normalização dos outros dados
    X_scaled = StandardScaler().fit_transform(X)
    
    np.savez(caminho_completo, X=X_scaled, y=y)
    print(f"[TESTE] Arquivo salvo em: {caminho_completo}")
    return caminho_completo

def salvar_dataset(X, y, nome_arquivo):
    """
    Salva o dataset em formato .npy (NumPy) para leitura rápida.
    X: Dados (features)
    y: Labels (ground truth para cálculo de Rand Index)
    """
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        
    caminho_completo = os.path.join(OUTPUT_DIR, nome_arquivo)
    # Normalização é sugerida no código de exemplo do sklearn para facilitar parâmetros
    X_scaled = StandardScaler().fit_transform(X)
    
    np.savez(caminho_completo, X=X_scaled, y=y)
    print(f"[GERADO] {nome_arquivo} | Shape: {X.shape}")

def gerar_sklearn_toy_data():
    """
    Gera 30 datasets baseados no exemplo do Scikit-Learn:
    - Noisy Circles, Moons, Blobs, No Structure, Anisotropic, Varied Variances.
    - 5 variações para cada tipo.
    Referência:
    """
    print(f"\n--- Gerando Dados Tipo Scikit-Learn ({N_SAMPLES} amostras) ---")
    
    rng = np.random.RandomState(SEED_BASE)
    
    # Loop para gerar 5 variações de cada tipo
    for i in range(1, 6):
        seed_iter = SEED_BASE + i
        
        # 1. Noisy Circles
        # Variação: Fator de escala e ruído
        factor = 0.3 + (i * 0.1) # 0.4 a 0.8
        noise = 0.03 + (i * 0.01)
        X, y = datasets.make_circles(n_samples=N_SAMPLES, factor=factor, noise=noise, random_state=seed_iter)
        salvar_dataset(X, y, f"sklearn_circles_v{i}.npz")

        # 2. Noisy Moons
        # Variação: Ruído
        noise = 0.05 + (i * 0.02)
        X, y = datasets.make_moons(n_samples=N_SAMPLES, noise=noise, random_state=seed_iter)
        salvar_dataset(X, y, f"sklearn_moons_v{i}.npz")

        # 3. Blobs (Simples)
        # Variação: Número de centros e aleatoriedade
        n_centers = 3 + (i % 3) # Varia entre 3, 4, 5 centros
        X, y = datasets.make_blobs(n_samples=N_SAMPLES, centers=n_centers, random_state=seed_iter)
        salvar_dataset(X, y, f"sklearn_blobs_v{i}.npz")

        # 4. No Structure (Ruído Uniforme)
        # Variação: Apenas a semente aleatória muda a distribuição dos pontos
        X_rand = np.random.RandomState(seed_iter).rand(N_SAMPLES, 2)
        y_rand = np.zeros(N_SAMPLES) # Sem classe definida
        salvar_dataset(X_rand, y_rand, f"sklearn_no_structure_v{i}.npz")

        # 5. Anisotropic (Blobs esticados)
        # Variação: Matriz de transformação (rotação e esticamento)
        X_base, y_base = datasets.make_blobs(n_samples=N_SAMPLES, centers=3, random_state=seed_iter)
        # Cria uma transformação linear diferente a cada iteração
        theta = np.radians(15 * i)
        stretch_x = 0.6 + (0.1 * i)
        stretch_y = 0.4 - (0.05 * i)
        transformation = [[np.cos(theta)*stretch_x, -np.sin(theta)*stretch_y], 
                          [np.sin(theta)*stretch_x, np.cos(theta)*stretch_y]]
        X_aniso = np.dot(X_base, transformation)
        salvar_dataset(X_aniso, y_base, f"sklearn_aniso_v{i}.npz")

        # 6. Varied Variances (Blobs com desvios padrão diferentes)
        # Variação: Lista de desvios padrão
        std_devs = [1.0, 2.5 - (0.3*i), 0.5 + (0.1*i)]
        X_varied, y_varied = datasets.make_blobs(n_samples=N_SAMPLES, cluster_std=std_devs, random_state=seed_iter)
        salvar_dataset(X_varied, y_varied, f"sklearn_varied_v{i}.npz")

def gerar_multivariada_normal():
    """
    Gera 10 datasets usando Distribuição Normal Multivariada.
    Controla: Sobreposição (overlap) e Formato (elíptico vs circular).
    Referência:
    """
    print(f"\n--- Gerando Dados Multivariados Customizados ({N_SAMPLES} amostras) ---")
    
    np.random.seed(SEED_BASE)
    
    # 5 Datasets: Clusters Circulares com variação de sobreposição
    for i in range(1, 6):
        # Quanto maior o i, maior a dispersão (sigma) -> Maior sobreposição
        # Centros fixos para comparar o efeito da dispersão
        means = [[0, 0], [4, 4], [0, 5]]
        cov_val = 0.5 + (i * 0.4) # Aumenta a variância
        cov = [[cov_val, 0], [0, cov_val]] # Matriz diagonal (circular)
        
        X_list = []
        y_list = []
        n_per_cluster = N_SAMPLES // 3
        
        for idx, mean in enumerate(means):
            x_c = np.random.multivariate_normal(mean, cov, n_per_cluster)
            X_list.append(x_c)
            y_list.append(np.full(n_per_cluster, idx))
            
        X = np.vstack(X_list)
        y = np.concatenate(y_list)
        salvar_dataset(X, y, f"custom_multivar_circular_overlap_v{i}.npz")

    # 5 Datasets: Clusters Elípticos (Alongados)
    for i in range(1, 6):
        means = [[0, 0], [5, 5], [5, -2]]
        
        X_list = []
        y_list = []
        n_per_cluster = N_SAMPLES // 3
        
        for idx, mean in enumerate(means):
            # Criar covariância com correlação para gerar elipse
            # Varia a 'finesse' da elipse e a rotação baseada no indice i
            sigma_x = 1.0 + (i * 0.2)
            sigma_y = 0.3 # Eixo menor fixo para garantir alongamento
            rho = 0.5 + (0.05 * i) # Correlação
            
            # Matriz de covariância [[sx^2, rho*sx*sy], [rho*sx*sy, sy^2]]
            cov_xy = rho * sigma_x * sigma_y
            cov = [[sigma_x**2, cov_xy], [cov_xy, sigma_y**2]]
            
            x_c = np.random.multivariate_normal(mean, cov, n_per_cluster)
            X_list.append(x_c)
            y_list.append(np.full(n_per_cluster, idx))

        X = np.vstack(X_list)
        y = np.concatenate(y_list)
        salvar_dataset(X, y, f"custom_multivar_elliptical_v{i}.npz")

if __name__ == "__main__":
    gerar_sklearn_toy_data()   # Gera 30 arquivos
    gerar_multivariada_normal() # Gera 10 arquivos
    print(f"\nConcluído! Total de arquivos gerados em '{OUTPUT_DIR}': {len(os.listdir(OUTPUT_DIR))}")