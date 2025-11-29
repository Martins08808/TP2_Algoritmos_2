# import numpy as np
# import time
# import os
# from distances import gerar_matriz_distancias
# from algorithms import cluster_guloso, cluster_refinamento

# # Importa o gerador para garantir que o dado existe
# from data_generator import gerar_dataset_teste 

# # =============================================================================
# # 1. CONFIGURAÇÕES E CONSTANTES
# # =============================================================================

# # Escolha do Algoritmo: 'GULOSO' ou 'REFINAMENTO'
# ALGORITMO_ESCOLHIDO = 'GULOSO' 

# # Parâmetros do Clustering
# K = 3  # Número de centros (clusters) - Alinhado com o make_blobs do teste
# SEMENTE_ALEATORIA = 42 

# # Configuração da Métrica de Distância
# METRICA = 'minkowski' 
# P_MINKOWSKI = 2        

# # Caminho do arquivo de teste
# ARQUIVO_TESTE = "integration_test.npz"

# # =============================================================================
# # 2. CARREGAMENTO DO DATASET
# # =============================================================================

# def obter_dataset():
#     """
#     Tenta carregar o dataset de teste do disco.
#     Se não existir, chama o data_generator para criá-lo.
#     Retorna: Matriz X (N_amostras x N_features)
#     """
#     # 1. Garante que o arquivo existe
#     caminho_arquivo = gerar_dataset_teste(ARQUIVO_TESTE)
    
#     # 2. Carrega o arquivo .npz
#     try:
#         dados_carregados = np.load(caminho_arquivo)
#         X = dados_carregados['X'] # Recupera a matriz de features
#         # y = dados_carregados['y'] # Labels (não usados no clustering, mas úteis para validação futura)
        
#         print(f"-> Arquivo carregado: {caminho_arquivo}")
#         return X
        
#     except Exception as e:
#         print(f"ERRO CRÍTICO ao abrir o dataset: {e}")
#         exit(1)

# # =============================================================================
# # 3. EXECUÇÃO PRINCIPAL
# # =============================================================================

# def main():
#     print(f"=== EXECUÇÃO DO K-CENTRO (Integração com Dataset) ===\n")
#     print(f"Configuração: K={K} | Métrica={METRICA} (p={P_MINKOWSKI})")

#     # 1. Carregar Dados (Integração feita aqui)
#     dados = obter_dataset()
    
#     # Validação básica
#     n_pontos = dados.shape[0]
#     print(f"-> Dataset carregado na memória. Shape: {dados.shape}")

#     # 2. Calcular Matriz de Distâncias (Pré-processamento)
#     start_dist = time.time()
#     print(f"-> Calculando matriz de distâncias ({METRICA})...")
    
#     # Nota: dados já vêm normalizados do data_generator
#     matriz_dist = gerar_matriz_distancias(dados, METRICA, p=P_MINKOWSKI)
    
#     tempo_dist = time.time() - start_dist
#     print(f"   (Concluído em {tempo_dist:.4f}s)\n")

#     # 3. Executar o Algoritmo Escolhido
#     print(f"-> Executando algoritmo: {ALGORITMO_ESCOLHIDO}...")
#     start_alg = time.time()

#     centros_escolhidos = []
#     raio_final = 0.0

#     if ALGORITMO_ESCOLHIDO == 'GULOSO':
#         centros_escolhidos, raio_final = cluster_guloso(
#             matriz_dist, 
#             k=K, 
#             semente=SEMENTE_ALEATORIA
#         )
    
#     elif ALGORITMO_ESCOLHIDO == 'REFINAMENTO':
#         centros_escolhidos, raio_final = cluster_refinamento(
#             matriz_dist, 
#             k=K, 
#             razao_parada=0.01
#         )
        
#     else:
#         print(f"ERRO: Algoritmo '{ALGORITMO_ESCOLHIDO}' não reconhecido.")
#         return

#     tempo_alg = time.time() - start_alg

#     # 4. Exibir Resultados
#     print("\n" + "="*40)
#     print("RESULTADO FINAL")
#     print("="*40)
#     print(f"Algoritmo:       {ALGORITMO_ESCOLHIDO}")
#     print(f"Tempo Total:     {tempo_alg:.5f} s")
#     print("-" * 40)
#     print(f"RAIO MÁXIMO:     {raio_final:.5f}")
#     print(f"CENTROS (IDX):   {centros_escolhidos}")
#     print("="*40)

#     # Verifica se a quantidade de centros bate com K
#     if len(centros_escolhidos) == K:
#         print("[SUCESSO] O algoritmo retornou o número correto de centros.")
#     else:
#         print(f"[ALERTA] O algoritmo retornou {len(centros_escolhidos)} centros (Esperado: {K}).")

#     # Opcional: Mostrar coordenadas dos centros
#     print("\nCoordenadas dos Centros:")
#     for idx in centros_escolhidos:
#         print(f"  Índice {idx}: {dados[idx]}")

# if __name__ == "__main__":
#     main()


import numpy as np
import time
from distances import gerar_matriz_distancias
from algorithms import cluster_guloso, cluster_refinamento
from data_generator import gerar_dataset_teste 

# --- NOVO IMPORT ---
from metrics import atribuir_labels_clusters, calcular_metricas_validacao

# =============================================================================
# 1. CONFIGURAÇÕES
# =============================================================================
ALGORITMO_ESCOLHIDO = 'GULOSO' 
K = 3 
SEMENTE_ALEATORIA = 42 
METRICA = 'minkowski' 
P_MINKOWSKI = 2        
ARQUIVO_TESTE = "integration_test.npz"

# =============================================================================
# 2. CARREGAMENTO DO DATASET
# =============================================================================
def obter_dataset():
    """
    Retorna X (dados) e y (classes reais/ground truth).
    """
    caminho_arquivo = gerar_dataset_teste(ARQUIVO_TESTE)
    
    try:
        dados_carregados = np.load(caminho_arquivo)
        X = dados_carregados['X'] 
        y = dados_carregados['y'] # Agora estamos recuperando o Y também!
        
        print(f"-> Arquivo carregado: {caminho_arquivo}")
        return X, y
        
    except Exception as e:
        print(f"ERRO CRÍTICO: {e}")
        exit(1)

# =============================================================================
# 3. EXECUÇÃO PRINCIPAL
# =============================================================================
def main():
    print(f"=== EXECUÇÃO DO K-CENTRO (k={K}) ===\n")

    # 1. Carregar Dados e Labels Reais
    X, y_true = obter_dataset()
    n_pontos = X.shape[0]

    # 2. Calcular Matriz de Distâncias
    start_dist = time.time()
    print(f"-> Calculando matriz de distâncias ({METRICA})...")
    matriz_dist = gerar_matriz_distancias(X, METRICA, p=P_MINKOWSKI)
    tempo_dist = time.time() - start_dist
    print(f"   (Matriz calculada em {tempo_dist:.4f}s)\n")

    # 3. Executar Algoritmo
    print(f"-> Executando algoritmo: {ALGORITMO_ESCOLHIDO}...")
    start_alg = time.time()

    centros_indices = []
    raio_final = 0.0

    if ALGORITMO_ESCOLHIDO == 'GULOSO':
        centros_indices, raio_final = cluster_guloso(matriz_dist, k=K, semente=SEMENTE_ALEATORIA)
    elif ALGORITMO_ESCOLHIDO == 'REFINAMENTO':
        centros_indices, raio_final = cluster_refinamento(matriz_dist, k=K, razao_parada=0.01)

    tempo_alg = time.time() - start_alg

    # 4. Calcular Métricas de Qualidade (NOVA ETAPA)
    # Primeiro: Atribuir cada ponto ao centro mais próximo
    labels_preditos = atribuir_labels_clusters(matriz_dist, centros_indices)
    
    # Segundo: Calcular Silhouette e ARI
    metricas = calcular_metricas_validacao(matriz_dist, labels_preditos, y_true)

    # 5. Exibir Resultados
    print("\n" + "="*40)
    print("RESULTADO FINAL")
    print("="*40)
    print(f"Algoritmo:       {ALGORITMO_ESCOLHIDO}")
    print(f"Tempo Execução:  {tempo_alg:.5f} s")
    print("-" * 40)
    print(f"Raio Máximo:     {raio_final:.5f}")
    print(f"Silhouette:      {metricas['Silhouette']:.5f}")
    print(f"Adj. Rand Index: {metricas['ARI']:.5f}")
    print("-" * 40)
    print(f"Centros (Idx):   {centros_indices}")
    print("="*40)

if __name__ == "__main__":
    main()