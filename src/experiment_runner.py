import numpy as np
import pandas as pd
import time
import os
import glob
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, adjusted_rand_score

# Imports do Projeto
from distances import gerar_matriz_distancias
from algorithms import cluster_guloso, cluster_refinamento
from data_loader import carregar_dados_sinteticos, carregar_dados_german, carregar_dados_reais

# =============================================================================
# CONFIGURAÇÕES
# =============================================================================
N_REPETICOES = 15
LARGURAS_REFINAMENTO = [0.01, 0.05, 0.10, 0.15, 0.25]

# --- CONFIGURAÇÃO DE FILTRO ---
# DATA_DIR = "synthetic"

ALEMAO = "SouthGermanCredit_LIMPO.csv"

DATASETS_SINTETICOS = [
    'multivar_01_circular_sem_overlap.csv',
    'multivar_02_eliptico_sem_overlap.csv',
    'multivar_03_circular_baixa_overlap.csv',
    'multivar_04_eliptico_baixa_overlap.csv',
    'multivar_05_circular_media_overlap.csv',
    'multivar_06_eliptico_media_overlap.csv',
    'multivar_07_circular_alta_overlap.csv',
    'multivar_08_eliptico_alta_overlap.csv',
    'multivar_09_circular_correlacao_positiva.csv',
    'multivar_10_eliptico_tamanhos_diferentes.csv',

    # 'sklearn_01_circles_v1.csv',
    # 'sklearn_01_circles_v2.csv',
    # 'sklearn_01_circles_v3.csv',
    # 'sklearn_01_circles_v4.csv',
    # 'sklearn_01_circles_v5.csv',

    # 'sklearn_02_moons_v1.csv',
    # 'sklearn_02_moons_v2.csv',
    # 'sklearn_02_moons_v3.csv',
    # 'sklearn_02_moons_v4.csv',
    # 'sklearn_02_moons_v5.csv',

    # 'sklearn_03_blobs_v1.csv',
    # 'sklearn_03_blobs_v2.csv',
    # 'sklearn_03_blobs_v3.csv',
    # 'sklearn_03_blobs_v4.csv',
    # 'sklearn_03_blobs_v5.csv',

    # 'sklearn_04_no_structure_v1.csv',
    # 'sklearn_04_no_structure_v2.csv',
    # 'sklearn_04_no_structure_v3.csv',
    # 'sklearn_04_no_structure_v4.csv',
    # 'sklearn_04_no_structure_v5.csv',

    # 'sklearn_05_aniso_v1.csv',
    # 'sklearn_05_aniso_v2.csv',
    # 'sklearn_05_aniso_v3.csv',
    # 'sklearn_05_aniso_v4.csv',
    # 'sklearn_05_aniso_v5.csv',

    # 'sklearn_06_varied_v1.csv',
    # 'sklearn_06_varied_v2.csv',
    # 'sklearn_06_varied_v3.csv',
    # 'sklearn_06_varied_v4.csv',
    # 'sklearn_06_varied_v5.csv',
]


# DATASETS_ALVO = DATASETS_SINTETICOS

DATA_DIR = "data"
DATASETS_ALVO = [
    # 'Iranian Customer Churn_LIMPO.csv',
    'winequality-red_LIMPO.csv',
    'winequality-white_LIMPO.csv',
]    

# =============================================================================
# FUNÇÕES AUXILIARES
# =============================================================================

def atribuir_labels_clusters(matriz_dist, indices_centros):
    """
    Dada a matriz de distâncias e os centros escolhidos,
    atribui cada ponto ao centro mais próximo (Labeling).
    
    Retorna:
        labels_pred: Array (N,) com o índice do cluster (0 a k-1) para cada ponto.
    """
    sub_matriz = matriz_dist[:, indices_centros]
    labels_pred = np.argmin(sub_matriz, axis=1)
    
    return labels_pred

def rodar_kmeans(X, k, y_true):
    """ Executa K-Means (Baseline) """
    start = time.time()
    # K-Means usa distância Euclidiana por padrão na função objetivo
    kmeans = KMeans(n_clusters=k, init='random', n_init=1)
    kmeans.fit(X)
    tempo = time.time() - start
    
    X_dist = kmeans.transform(X) 
    raio_max = np.max(np.min(X_dist, axis=1))
    
    # Cálculo direto das métricas para o K-Means
    try:
        # Para K-Means, geralmente avaliamos a geometria Euclidiana
        sil = silhouette_score(X, kmeans.labels_, metric='euclidean')
    except Exception as e:
        print(f"      [AVISO] Erro Silhouette K-Means: {e}")
        sil = -1.0
        
    ari = adjusted_rand_score(y_true, kmeans.labels_)
    
    return {'Raio': raio_max, 'Silhouette': sil, 'ARI': ari, 'Tempo': tempo}

def calcular_raio_real(matriz_dist, centros, labels):
    """Calcula o raio máximo (makespan) da solução usando a matriz de distâncias correta."""
    raio_max = 0.0
    n_pontos = matriz_dist.shape[0]
    for i in range(n_pontos):
        centro_do_ponto = centros[labels[i]]
        dist = matriz_dist[i, centro_do_ponto]
        if dist > raio_max:
            raio_max = dist
    return raio_max


def executar_ciclo(algoritmo_nome, func_algo, matriz_dist, y_true, k_clusters, params_extras):
    """ 
    Roda N repetições para um algoritmo específico calculando métricas diretamente.

    Agora usa:
        → raio real (calcular_raio_real)
    em vez do raio retornado pelo algoritmo.
    """
    resultados = []
    nome_metrica = params_extras.get('metrica_nome', 'N/A')
    largura = params_extras.get('largura', 'N/A')

    for i in range(N_REPETICOES):

        seed = int(time.time() * 1000) % 10000 + i
        start = time.time()

        try:
            # 1. Execução do algoritmo
            if algoritmo_nome == 'GULOSO':
                centros, _ = func_algo(matriz_dist, k_clusters, semente=seed)

            elif algoritmo_nome == 'REFINAMENTO':
                np.random.seed(seed)
                centros, _ = func_algo(
                    matriz_dist, 
                    k_clusters, 
                    razao_parada=params_extras['largura']
                )

            tempo = time.time() - start

            # 2. Atribuir rótulos
            labels = atribuir_labels_clusters(matriz_dist, centros)

            # 3. Calcular métricas
            
            # ---- Raio real (CORRIGIDO) ----
            raio_real = calcular_raio_real(matriz_dist, centros, labels)

            # Ajusted Rand Index
            ari = adjusted_rand_score(y_true, labels)

            # Silhouette
            try:
                if len(np.unique(labels)) > 1:
                    sil = silhouette_score(matriz_dist, labels, metric='precomputed')
                else:
                    sil = -1.0
            except:
                sil = -1.0

            resultados.append({
                'Algoritmo': algoritmo_nome,
                'Metrica_Dist': nome_metrica,
                'Parametro_Largura': largura,
                'Repeticao': i + 1,
                'Raio': raio_real,             # <--- AGORA O RAIO REAL
                'Silhouette': sil,
                'ARI': ari,
                'Tempo': tempo
            })

        except Exception as e:
            print(f"[ERRO] Falha na repetição {i}: {e}")

    return resultados


# =============================================================================
# MAIN LOOP
# =============================================================================

def main():
    # 1. Busca arquivos SINTÉTICOS na pasta padrão
    todos_arquivos = glob.glob(os.path.join(DATA_DIR, "*.csv"))
    todos_arquivos.sort()
    
    # 2. Constrói a lista final de arquivos para processar
    arquivos_para_processar = []
    
    # --- LÓGICA MISTA (Sintéticos + German) ---
    if len(DATASETS_ALVO) > 0:
        print(f"=== MODO FILTRADO: Buscando alvos ===")
        
        for alvo in DATASETS_ALVO:
            if 'SouthGerman' in alvo:
                arquivos_para_processar.append(alvo)
                print(f"  [+] Alvo Especial detectado: {alvo}")
                continue
            
            encontrado = False
            for arq_real in todos_arquivos:
                if os.path.basename(arq_real) == alvo:
                    arquivos_para_processar.append(arq_real)
                    encontrado = True
                    break
            if not encontrado:
                print(f"  [AVISO] Dataset sintético '{alvo}' não encontrado em '{DATA_DIR}'.")
    else:
        arquivos_para_processar = todos_arquivos

    if not arquivos_para_processar:
        print("\n[ERRO] Nenhum arquivo para processar.")
        return

    print(f"Total de arquivos na fila: {len(arquivos_para_processar)}\n")

    os.makedirs("resultados", exist_ok=True)

    for arquivo in arquivos_para_processar:
        nome_dataset = os.path.basename(arquivo)
        print(f"--> Processando: {nome_dataset}")
        
        # =====================================================================
        # CARREGAMENTO DOS DADOS
        # =====================================================================
        try:
            if 'SouthGerman' in nome_dataset:
                print("    [LOADER] Usando carregador específico para German Credit...")
                X, y_true = carregar_dados_german()
            else:
                # X, y_true = carregar_dados_sinteticos(arquivo)
                X, y_true = carregar_dados_reais(arquivo)
            
            k_dinamico = len(np.unique(y_true))
            print(f"    Shape: {X.shape} | Classes (K): {k_dinamico}")
            
        except Exception as e:
            print(f"    [ERRO] Falha ao carregar {nome_dataset}: {e}")
            continue
        
        # Gerar Matrizes
        print("    Gerando matrizes de distância...")
        metricas_map = []
        
        try:
            m_man = gerar_matriz_distancias(X, 'minkowski', p=1)
            metricas_map.append(('Manhattan', m_man))
        except: pass

        try:
            m_euc = gerar_matriz_distancias(X, 'minkowski', p=2)
            metricas_map.append(('Euclidiana', m_euc))
        except: pass

        try:
            m_mah = gerar_matriz_distancias(X, 'mahalanobis')
            metricas_map.append(('Mahalanobis', m_mah))
        except:
            print("    [INFO] Mahalanobis ignorada (Singular)")

        # Execução dos Algoritmos
        resultados_dataset = []

        # GULOSO
        for nome_met, matriz in metricas_map:
            res = executar_ciclo('GULOSO', cluster_guloso, matriz, y_true, k_dinamico, {'metrica_nome': nome_met})
            resultados_dataset.extend(res)

        # REFINAMENTO
        for nome_met, matriz in metricas_map:
            for largura in LARGURAS_REFINAMENTO:
                res = executar_ciclo('REFINAMENTO', cluster_refinamento, matriz, y_true, k_dinamico, 
                                     {'metrica_nome': nome_met, 'largura': largura})
                resultados_dataset.extend(res)

        # BASELINE K-MEANS
        for i in range(N_REPETICOES):
            dados_km = rodar_kmeans(X, k_dinamico, y_true)
            resultados_dataset.append({
                'Algoritmo': 'K-MEANS',
                'Metrica_Dist': 'Euclidiana',
                'Parametro_Largura': 'N/A',
                'Repeticao': i + 1,
                'Raio': dados_km['Raio'],
                'Silhouette': dados_km['Silhouette'],
                'ARI': dados_km['ARI'],
                'Tempo': dados_km['Tempo']
            })

        # =============================================================================
        # SALVAMENTO
        # =============================================================================
        if resultados_dataset:
            for r in resultados_dataset:
                r['Dataset'] = nome_dataset
            
            df_final = pd.DataFrame(resultados_dataset)
            
            cols = ['Dataset', 'Algoritmo', 'Metrica_Dist', 'Parametro_Largura', 
                    'Repeticao', 'Raio', 'Silhouette', 'ARI', 'Tempo']
            df_final = df_final[cols]
            
            nome_sem_extensao = os.path.splitext(nome_dataset)[0]
            caminho_saida = f"resultados/result_{nome_sem_extensao}.csv"
            
            df_final.to_csv(caminho_saida, index=False)
            print(f"    [OK] Resultados salvos em: {caminho_saida}")
        else:
            print(f"    [AVISO] Nenhum resultado gerado para {nome_dataset}.")

    print("\n=== TODOS OS PROCESSOS FINALIZADOS ===")

if __name__ == "__main__":
    main()