import numpy as np
import pandas as pd
import time
import os
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin_min
from sklearn.preprocessing import StandardScaler, MinMaxScaler # <--- IMPORT NOVO

# Imports do Projeto (Mantidos)
from distances import gerar_matriz_distancias
from algorithms import cluster_guloso, cluster_refinamento
from metrics import atribuir_labels_clusters, calcular_metricas_validacao
# from data_generator import gerar_dataset_teste  <-- NÃO PRECISA MAIS DESTE

# =============================================================================
# CONFIGURAÇÕES GERAIS DO EXPERIMENTO
# =============================================================================
N_REPETICOES = 15
# O dataset de crédito geralmente tem 2 classes (Bom/Ruim), então K=2 é o ideal para comparação
K_CLUSTERS = 2 
DATASET_FILE = "SouthGermanCredit_LIMPO.csv"

# Variações de Largura para o Refinamento (1% a 25%)
LARGURAS_REFINAMENTO = [0.01, 0.05, 0.10, 0.15, 0.25]

# =============================================================================
# FUNÇÕES AUXILIARES
# =============================================================================

def carregar_dados():
    """
    Lê o CSV, separa features e target, e normaliza os dados.
    """
    if not os.path.exists(DATASET_FILE):
        raise FileNotFoundError(f"Arquivo {DATASET_FILE} não encontrado!")

    # 1. Ler o CSV
    df = pd.read_csv(DATASET_FILE)
    
    # 2. Separar X (features) e y (target)
    # Assumindo que 'credit_risk' é a coluna alvo conforme seu snippet
    if 'credit_risk' in df.columns:
        y = df['credit_risk'].values
        X_raw = df.drop(columns=['credit_risk']).values
    else:
        # Fallback caso o nome da coluna mude, assume que é a última
        y = df.iloc[:, -1].values
        X_raw = df.iloc[:, :-1].values

    # 3. Normalização (CRÍTICO PARA CLUSTERING DE DADOS HETEROGÊNEOS)
    # Escolha entre StandardScaler (Média 0, Desvio 1) ou MinMaxScaler (0 a 1)
    # Para Mahalanobis/Euclidiana, StandardScaler geralmente é mais robusto a outliers leves
    scaler = StandardScaler()
    X = scaler.fit_transform(X_raw)
    
    print(f"   -> Dados carregados: {X.shape[0]} amostras, {X.shape[1]} atributos.")
    return X, y

def rodar_kmeans(X, k, y_true):
    """
    Executa K-Means (Baseline) e retorna métricas no mesmo formato.
    """
    start = time.time()
    kmeans = KMeans(n_clusters=k, init='random', n_init=1)
    kmeans.fit(X)
    tempo = time.time() - start
    
    X_dist = kmeans.transform(X) 
    dists_pontos = np.min(X_dist, axis=1)
    raio_max = np.max(dists_pontos)
    
    from sklearn.metrics import silhouette_score, adjusted_rand_score
    try:
        # Silhouette pode ser pesada se o dataset for muito grande (>10k linhas)
        sil = silhouette_score(X, kmeans.labels_, metric='euclidean')
    except:
        sil = -1.0
    ari = adjusted_rand_score(y_true, kmeans.labels_)
    
    return {
        'Raio': raio_max,
        'Silhouette': sil,
        'ARI': ari,
        'Tempo': tempo
    }

def executar_ciclo_algoritmo(algoritmo_nome, func_algo, matriz_dist, X_dados, y_true, params_extras={}):
    resultados_ciclo = []
    
    nome_metrica = params_extras.get('metrica_nome', 'N/A')
    param_largura = params_extras.get('largura', 'N/A')
    
    print(f"   -> Rodando {algoritmo_nome} | Dist: {nome_metrica} | Param: {param_largura}...")

    for i in range(N_REPETICOES):
        seed = int(time.time() * 1000) % 10000 + i
        
        start = time.time()
        
        if algoritmo_nome == 'GULOSO':
            centros, raio = func_algo(matriz_dist, K_CLUSTERS, semente=seed)
        elif algoritmo_nome == 'REFINAMENTO':
            np.random.seed(seed)
            razao = params_extras['largura']
            centros, raio = func_algo(matriz_dist, K_CLUSTERS, razao_parada=razao)
            
        tempo = time.time() - start
        
        labels = atribuir_labels_clusters(matriz_dist, centros)
        stats = calcular_metricas_validacao(matriz_dist, labels, y_true)
        
        resultados_ciclo.append({
            'Dataset': DATASET_FILE,
            'Algoritmo': algoritmo_nome,
            'Metrica_Dist': nome_metrica,
            'Parametro_Largura': param_largura,
            'Repeticao': i + 1,
            'Raio': raio,
            'Silhouette': stats['Silhouette'],
            'ARI': stats['ARI'],
            'Tempo': tempo
        })
        
    return resultados_ciclo

# =============================================================================
# MAIN
# =============================================================================

def main():
    print(f"=== INICIANDO EXPERIMENTO (Dataset: {DATASET_FILE}) ===\n")
    
    # 1. Carregar Dados do CSV
    try:
        X, y_true = carregar_dados()
    except Exception as e:
        print(f"Erro ao carregar dados: {e}")
        return

    registros = []
    
    # -------------------------------------------------------------------------
    # PRÉ-CÁLCULO DAS MATRIZES DE DISTÂNCIA
    # -------------------------------------------------------------------------
    print("1. Pré-calculando matrizes de distância...")
    # Atenção: Matrizes N x N podem consumir muita RAM se N > 5000
    
    t0 = time.time()
    matriz_manhattan = gerar_matriz_distancias(X, 'minkowski', p=1)
    print(f"   - Manhattan pronta ({time.time()-t0:.2f}s)")
    
    t0 = time.time()
    matriz_euclidiana = gerar_matriz_distancias(X, 'minkowski', p=2)
    print(f"   - Euclidiana pronta ({time.time()-t0:.2f}s)")
    
    # Mahalanobis pode ser instável se houver colinearidade perfeita, 
    # mas o StandardScaler ajuda.
    try:
        t0 = time.time()
        matriz_mahalanobis = gerar_matriz_distancias(X, 'mahalanobis')
        print(f"   - Mahalanobis pronta ({time.time()-t0:.2f}s)")
        usar_mahalanobis = True
    except Exception as e:
        print(f"   - Mahalanobis FALHOU: {e}. Ignorando esta métrica.")
        usar_mahalanobis = False
    
    metricas_map = [
        ('Manhattan (p=1)', matriz_manhattan),
        ('Euclidiana (p=2)', matriz_euclidiana)
    ]
    if usar_mahalanobis:
        metricas_map.append(('Mahalanobis', matriz_mahalanobis))

    # -------------------------------------------------------------------------
    # 2. ALGORITMO GULOSO
    # -------------------------------------------------------------------------
    print("\n2. Executando Algoritmo GULOSO...")
    for nome_met, matriz in metricas_map:
        res = executar_ciclo_algoritmo(
            'GULOSO', 
            cluster_guloso, 
            matriz, X, y_true, 
            params_extras={'metrica_nome': nome_met}
        )
        registros.extend(res)

    # -------------------------------------------------------------------------
    # 3. ALGORITMO REFINAMENTO
    # -------------------------------------------------------------------------
    print("\n3. Executando Algoritmo REFINAMENTO...")
    for nome_met, matriz in metricas_map:
        for largura in LARGURAS_REFINAMENTO:
            res = executar_ciclo_algoritmo(
                'REFINAMENTO', 
                cluster_refinamento, 
                matriz, X, y_true, 
                params_extras={'metrica_nome': nome_met, 'largura': largura}
            )
            registros.extend(res)

    # -------------------------------------------------------------------------
    # 4. ALGORITMO K-MEANS (Baseline)
    # -------------------------------------------------------------------------
    print("\n4. Executando Baseline K-MEANS...")
    for i in range(N_REPETICOES):
        dados_km = rodar_kmeans(X, K_CLUSTERS, y_true)
        
        registros.append({
            'Dataset': DATASET_FILE,
            'Algoritmo': 'K-MEANS',
            'Metrica_Dist': 'Euclidiana (Nativo)',
            'Parametro_Largura': 'N/A',
            'Repeticao': i + 1,
            'Raio': dados_km['Raio'],
            'Silhouette': dados_km['Silhouette'],
            'ARI': dados_km['ARI'],
            'Tempo': dados_km['Tempo']
        })

    # -------------------------------------------------------------------------
    # 5. SALVAR RESULTADOS
    # -------------------------------------------------------------------------
    df_res = pd.DataFrame(registros)
    
    OUTPUT_DIR = "resultados"
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    dataset_name = os.path.splitext(DATASET_FILE)[0]
    timestamp = int(time.time())
    nome_arquivo = f"resultado_{dataset_name}_{timestamp}.csv"
    caminho_completo = os.path.join(OUTPUT_DIR, nome_arquivo)
    
    df_res.to_csv(caminho_completo, index=False)
    
    print("\n" + "="*50)
    print("EXPERIMENTO CONCLUÍDO!")
    print(f"Salvo em: {caminho_completo}")
    print("="*50)

if __name__ == "__main__":
    main()