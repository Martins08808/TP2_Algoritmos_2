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
from metrics import atribuir_labels_clusters, calcular_metricas_validacao
from data_loader import carregar_dados_sinteticos

# =============================================================================
# CONFIGURAÇÕES
# =============================================================================
N_REPETICOES = 15
LARGURAS_REFINAMENTO = [0.01, 0.05, 0.10, 0.15, 0.25]

# --- CONFIGURAÇÃO DE FILTRO ---
DATA_DIR = "synthetic"
DATASETS_ALVO = [
    'sklearn_01_circles_v1.csv',
]

# =============================================================================
# FUNÇÕES AUXILIARES
# =============================================================================

def rodar_kmeans(X, k, y_true):
    """ Executa K-Means (Baseline) """
    start = time.time()
    kmeans = KMeans(n_clusters=k, init='random', n_init=1)
    kmeans.fit(X)
    tempo = time.time() - start
    
    X_dist = kmeans.transform(X) 
    raio_max = np.max(np.min(X_dist, axis=1))
    
    try:
        sil = silhouette_score(X, kmeans.labels_, metric='euclidean')
    except:
        sil = -1.0
        
    ari = adjusted_rand_score(y_true, kmeans.labels_)
    
    return {'Raio': raio_max, 'Silhouette': sil, 'ARI': ari, 'Tempo': tempo}

def executar_ciclo(algoritmo_nome, func_algo, matriz_dist, y_true, k_clusters, params_extras):
    """ Roda N repetições para um algoritmo específico """
    resultados = []
    nome_metrica = params_extras.get('metrica_nome', 'N/A')
    largura = params_extras.get('largura', 'N/A')

    for i in range(N_REPETICOES):
        seed = int(time.time() * 1000) % 10000 + i
        start = time.time()
        
        try:
            if algoritmo_nome == 'GULOSO':
                centros, raio = func_algo(matriz_dist, k_clusters, semente=seed)
            elif algoritmo_nome == 'REFINAMENTO':
                np.random.seed(seed)
                centros, raio = func_algo(matriz_dist, k_clusters, razao_parada=params_extras['largura'])
            
            tempo = time.time() - start
            labels = atribuir_labels_clusters(matriz_dist, centros)
            stats = calcular_metricas_validacao(matriz_dist, labels, y_true)
            
            resultados.append({
                'Algoritmo': algoritmo_nome,
                'Metrica_Dist': nome_metrica,
                'Parametro_Largura': largura,
                'Repeticao': i + 1,
                'Raio': raio,
                'Silhouette': stats['Silhouette'],
                'ARI': stats['ARI'],
                'Tempo': tempo
            })
        except Exception as e:
            print(f"      [ERRO] Falha na repetição {i}: {e}")

    return resultados

# =============================================================================
# MAIN LOOP
# =============================================================================

def main():
    # 1. Busca TODOS os arquivos
    todos_arquivos = glob.glob(os.path.join(DATA_DIR, "*.csv"))
    todos_arquivos.sort()
    
    # 2. Aplica o filtro de DATASETS_ALVO
    arquivos_para_processar = []
    
    if len(DATASETS_ALVO) > 0:
        print(f"=== MODO FILTRADO: Rodando apenas {len(DATASETS_ALVO)} datasets escolhidos ===")
        for arq in todos_arquivos:
            nome_base = os.path.basename(arq)
            if nome_base in DATASETS_ALVO:
                arquivos_para_processar.append(arq)
            
        encontrados = [os.path.basename(x) for x in arquivos_para_processar]
        for alvo in DATASETS_ALVO:
            if alvo not in encontrados:
                print(f"[AVISO] O dataset '{alvo}' está na lista de alvos mas NÃO foi encontrado na pasta '{DATA_DIR}'.")
    else:
        print("=== MODO COMPLETO: Rodando todos os datasets encontrados ===")
        arquivos_para_processar = todos_arquivos

    if not arquivos_para_processar:
        print("\n[ERRO] Nenhum arquivo para processar. Verifique a lista DATASETS_ALVO ou a pasta data.")
        return

    print(f"Total de arquivos na fila: {len(arquivos_para_processar)}\n")

    # Garante que a pasta de resultados existe antes de começar
    os.makedirs("resultados", exist_ok=True)

    for arquivo in arquivos_para_processar:
        nome_dataset = os.path.basename(arquivo)
        print(f"--> Processando: {nome_dataset}")
        
        # Carregar Dados
        try:
            X, y_true = carregar_dados_sinteticos(arquivo)
            k_dinamico = len(np.unique(y_true))
            print(f"    Shape: {X.shape} | Classes (K): {k_dinamico}")
        except Exception as e:
            print(f"    [ERRO CRÍTICO] Falha ao carregar {nome_dataset}: {e}")
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
        # SALVAMENTO INDIVIDUAL (ALTERADO)
        # =============================================================================
        if resultados_dataset:
            # Adiciona o nome do dataset em cada linha
            for r in resultados_dataset:
                r['Dataset'] = nome_dataset
            
            df_final = pd.DataFrame(resultados_dataset)
            
            cols = ['Dataset', 'Algoritmo', 'Metrica_Dist', 'Parametro_Largura', 
                    'Repeticao', 'Raio', 'Silhouette', 'ARI', 'Tempo']
            df_final = df_final[cols]
            
            # Remove a extensão .csv do nome original para criar o nome do arquivo de saída
            nome_sem_extensao = os.path.splitext(nome_dataset)[0]
            caminho_saida = f"resultados/result_{nome_sem_extensao}.csv"
            
            df_final.to_csv(caminho_saida, index=False)
            print(f"    [OK] Resultados salvos em: {caminho_saida}")
        else:
            print(f"    [AVISO] Nenhum resultado gerado para {nome_dataset}.")

    print("\n=== TODOS OS PROCESSOS FINALIZADOS ===")

if __name__ == "__main__":
    main()