import numpy as np
import pandas as pd
import time
import os  # <--- IMPORT ADICIONADO
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin_min

# Imports do Projeto (Assumindo que os arquivos existem no mesmo diretório)
from distances import gerar_matriz_distancias
from algorithms import cluster_guloso, cluster_refinamento
from metrics import atribuir_labels_clusters, calcular_metricas_validacao
from data_generator import gerar_dataset_teste

# =============================================================================
# CONFIGURAÇÕES GERAIS DO EXPERIMENTO
# =============================================================================
N_REPETICOES = 15
K_CLUSTERS = 3
DATASET_FILE = "integration_test.npz"

# Variações de Largura para o Refinamento (1% a 25%)
LARGURAS_REFINAMENTO = [0.01, 0.05, 0.10, 0.15, 0.25]

# =============================================================================
# FUNÇÕES AUXILIARES
# =============================================================================

def carregar_dados():
    # Garante que o dataset existe
    path = gerar_dataset_teste(DATASET_FILE)
    dados = np.load(path)
    return dados['X'], dados['y']

def rodar_kmeans(X, k, y_true):
    """
    Executa K-Means (Baseline) e retorna métricas no mesmo formato.
    """
    start = time.time()
    # n_init=1 pois faremos o loop de repetições manualmente fora
    kmeans = KMeans(n_clusters=k, init='random', n_init=1)
    kmeans.fit(X)
    tempo = time.time() - start
    
    # Calcula distância de cada ponto ao seu centro atribuído
    X_dist = kmeans.transform(X) 
    dists_pontos = np.min(X_dist, axis=1)
    raio_max = np.max(dists_pontos)
    
    # Métricas de Qualidade (ARI)
    # Silhouette calculada separadamente devido a peculiaridades de distância
    from sklearn.metrics import silhouette_score, adjusted_rand_score
    try:
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
    """
    Roda as 15 repetições para uma configuração específica de Algoritmo + Métrica.
    """
    resultados_ciclo = []
    
    nome_metrica = params_extras.get('metrica_nome', 'N/A')
    param_largura = params_extras.get('largura', 'N/A')
    
    print(f"   -> Rodando {algoritmo_nome} | Dist: {nome_metrica} | Param: {param_largura}...")

    for i in range(N_REPETICOES):
        seed = int(time.time() * 1000) % 10000 + i
        
        start = time.time()
        
        # Executa o algoritmo
        if algoritmo_nome == 'GULOSO':
            centros, raio = func_algo(matriz_dist, K_CLUSTERS, semente=seed)
        elif algoritmo_nome == 'REFINAMENTO':
            np.random.seed(seed)
            razao = params_extras['largura']
            centros, raio = func_algo(matriz_dist, K_CLUSTERS, razao_parada=razao)
            
        tempo = time.time() - start
        
        # Calcula Métricas de Qualidade
        labels = atribuir_labels_clusters(matriz_dist, centros)
        stats = calcular_metricas_validacao(matriz_dist, labels, y_true)
        
        # Registra linha
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
    print(f"=== INICIANDO EXPERIMENTO COMPLETO (Dataset: {DATASET_FILE}) ===\n")
    
    # 1. Carregar Dados
    X, y_true = carregar_dados()
    registros = []
    
    # -------------------------------------------------------------------------
    # PRÉ-CÁLCULO DAS MATRIZES DE DISTÂNCIA
    # -------------------------------------------------------------------------
    print("1. Pré-calculando matrizes de distância...")
    
    t0 = time.time()
    matriz_manhattan = gerar_matriz_distancias(X, 'minkowski', p=1)
    print(f"   - Manhattan pronta ({time.time()-t0:.2f}s)")
    
    t0 = time.time()
    matriz_euclidiana = gerar_matriz_distancias(X, 'minkowski', p=2)
    print(f"   - Euclidiana pronta ({time.time()-t0:.2f}s)")
    
    t0 = time.time()
    matriz_mahalanobis = gerar_matriz_distancias(X, 'mahalanobis')
    print(f"   - Mahalanobis pronta ({time.time()-t0:.2f}s)")
    
    # Dicionário para facilitar o loop
    metricas_map = [
        ('Manhattan (p=1)', matriz_manhattan),
        ('Euclidiana (p=2)', matriz_euclidiana),
        ('Mahalanobis', matriz_mahalanobis)
    ]

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
    print(f"   -> Rodando K-Means Padrão (Euclidiano)...")
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
    # 5. SALVAR RESULTADOS (ADAPTADO)
    # -------------------------------------------------------------------------
    df = pd.DataFrame(registros)
    
    # Definição do diretório de saída
    OUTPUT_DIR = "resultados"
    
    # Cria a pasta se ela não existir
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Extrai apenas o nome do arquivo sem extensão (ex: integration_test)
    dataset_name = os.path.splitext(DATASET_FILE)[0]
    timestamp = int(time.time())
    
    # Constrói o nome do arquivo: resultados/RESULTADO_dataset_timestamp.csv
    nome_arquivo = f"resultado_{dataset_name}_{timestamp}.csv"
    caminho_completo = os.path.join(OUTPUT_DIR, nome_arquivo)
    
    df.to_csv(caminho_completo, index=False)
    
    print("\n" + "="*50)
    print("EXPERIMENTO CONCLUÍDO COM SUCESSO!")
    print("="*50)
    print(f"Total de Execuções: {len(df)}")
    print(f"Arquivo Salvo em:   {caminho_completo}")
    print("\nResumo das Médias por Algoritmo:")
    print(df.groupby(['Algoritmo', 'Metrica_Dist'])[['Raio', 'Silhouette', 'ARI', 'Tempo']].mean())

if __name__ == "__main__":
    main()