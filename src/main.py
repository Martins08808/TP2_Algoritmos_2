import numpy as np
import time
from distances import gerar_matriz_distancias
from algorithms import cluster_guloso, cluster_refinamento

# =============================================================================
# 1. CONFIGURAÇÕES E CONSTANTES
# =============================================================================

# Escolha do Algoritmo: 'GULOSO' ou 'REFINAMENTO'
ALGORITMO_ESCOLHIDO = 'GULOSO' 

# Parâmetros do Clustering
K = 3  # Número de centros (clusters)
SEMENTE_ALEATORIA = 42  # Para reprodutibilidade

# Configuração da Métrica de Distância
METRICA = 'minkowski'  # Opções: 'minkowski' ou 'mahalanobis'
P_MINKOWSKI = 2        # 1 = Manhattan, 2 = Euclidiana (ignorado se Mahalanobis)

# =============================================================================
# 2. DEFINIÇÃO DO DATASET (DADO NO CÓDIGO)
# =============================================================================

def obter_dataset():
    """
    Retorna o dataset hardcoded ou gerado sinteticamente.
    Formato: Matriz Numpy (N_amostras x N_features)
    """
    # Exemplo: Dataset sintético com 3 grupos bem definidos para teste
    # Grupo 1: próximo a (0,0)
    # Grupo 2: próximo a (10,10)
    # Grupo 3: próximo a (20,5)
    np.random.seed(SEMENTE_ALEATORIA)
    
    grupo_1 = np.random.normal(loc=0.0, scale=1.0, size=(10, 2))
    grupo_2 = np.random.normal(loc=10.0, scale=1.0, size=(10, 2))
    grupo_3 = np.random.normal(loc=[20.0, 5.0], scale=1.0, size=(10, 2))
    
    dados = np.vstack([grupo_1, grupo_2, grupo_3])
    return dados

# =============================================================================
# 3. EXECUÇÃO PRINCIPAL
# =============================================================================

def main():
    print(f"=== EXECUÇÃO DO K-CENTRO (k={K}) ===\n")

    # 1. Carregar Dados
    dados = obter_dataset()
    n_pontos = dados.shape[0]
    print(f"-> Dataset carregado com {n_pontos} pontos.")

    # 2. Calcular Matriz de Distâncias (Pré-processamento)
    start_dist = time.time()
    print(f"-> Calculando matriz de distâncias ({METRICA})...")
    
    matriz_dist = gerar_matriz_distancias(dados, METRICA, p=P_MINKOWSKI)
    
    tempo_dist = time.time() - start_dist
    print(f"   (Concluído em {tempo_dist:.4f}s)\n")

    # 3. Executar o Algoritmo Escolhido
    print(f"-> Executando algoritmo: {ALGORITMO_ESCOLHIDO}...")
    start_alg = time.time()

    centros_escolhidos = []
    raio_final = 0.0

    if ALGORITMO_ESCOLHIDO == 'GULOSO':
        centros_escolhidos, raio_final = cluster_guloso(
            matriz_dist, 
            k=K, 
            semente=SEMENTE_ALEATORIA
        )
    
    elif ALGORITMO_ESCOLHIDO == 'REFINAMENTO':
        centros_escolhidos, raio_final = cluster_refinamento(
            matriz_dist, 
            k=K, 
            razao_parada=0.01
        )
        
    else:
        print(f"ERRO: Algoritmo '{ALGORITMO_ESCOLHIDO}' não reconhecido.")
        return

    tempo_alg = time.time() - start_alg

    # 4. Exibir Resultados (Formato esperado de saída)
    print("\n" + "="*40)
    print("RESULTADO FINAL")
    print("="*40)
    print(f"Algoritmo Utilizado: {ALGORITMO_ESCOLHIDO}")
    print(f"Tempo de Execução:   {tempo_alg:.5f} segundos")
    print("-" * 40)
    print(f"RAIO MÁXIMO (Custo): {raio_final:.5f}")
    print(f"CENTROS ESCOLHIDOS:  {centros_escolhidos}") # Índices dos pontos no dataset
    print("="*40)

    # Opcional: Mostrar coordenadas dos centros
    print("\nCoordenadas dos Centros:")
    for idx in centros_escolhidos:
        print(f"  Índice {idx}: {dados[idx]}")

if __name__ == "__main__":
    main()