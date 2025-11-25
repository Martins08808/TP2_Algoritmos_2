import numpy as np
from distances import gerar_matriz_distancias
from algorithms import cluster_guloso, cluster_refinamento

def reportar_teste(nome, passou, detalhe=""):
    status = "[SUCESSO]" if passou else "[FALHA]"
    print(f"{status} {nome} {detalhe}")

# =================================================================
# PREPARAÇÃO DOS DADOS (CENÁRIO ÓBVIO)
# =================================================================
# Grupo A: 5 pontos perto da origem (0,0)
grupo_a = np.random.rand(5, 2) 
# Grupo B: 5 pontos muito longe (100,100)
grupo_b = np.random.rand(5, 2) + 100 

# Junta tudo (índices 0-4 são Grupo A, índices 5-9 são Grupo B)
dados = np.vstack([grupo_a, grupo_b])
n_total = dados.shape[0]

print("--- Gerando Matriz de Distâncias (Euclidiana) ---")
matriz_D = gerar_matriz_distancias(dados, 'minkowski', p=2)

# =================================================================
# TESTE 1: ALGORITMO GULOSO (k=2)
# =================================================================
print("\n>>> Testando Algoritmo Guloso...")
centros_g, raio_g = cluster_guloso(matriz_D, k=2, semente=42)

# Verificação Lógica:
# Com k=2, ele DEVE pegar um ponto do grupo A (<5) e um do grupo B (>=5)
tem_grupo_a = any(c < 5 for c in centros_g)
tem_grupo_b = any(c >= 5 for c in centros_g)
tamanho_ok = len(centros_g) == 2

reportar_teste("Quantidade de Centros", tamanho_ok, f"(Obtido: {len(centros_g)})")
reportar_teste("Distribuição dos Grupos", tem_grupo_a and tem_grupo_b, 
               f"(Centros escolhidos: {centros_g})")

# O raio deve ser pequeno (apenas a variação interna do cluster, < 2.0 aprox), 
# não pode ser gigante (> 90.0) senão ele misturou os clusters.
raio_logico = raio_g < 10.0
reportar_teste("Qualidade do Raio", raio_logico, f"(Raio: {raio_g:.4f})")


# =================================================================
# TESTE 2: ALGORITMO REFINAMENTO (k=2)
# =================================================================
print("\n>>> Testando Algoritmo Refinamento (Busca Binária)...")
centros_r, raio_r = cluster_refinamento(matriz_D, k=2, razao_parada=0.01)

tem_grupo_a_r = any(c < 5 for c in centros_r)
tem_grupo_b_r = any(c >= 5 for c in centros_r)
tamanho_ok_r = len(centros_r) == 2

reportar_teste("Quantidade de Centros", tamanho_ok_r, f"(Obtido: {len(centros_r)})")
reportar_teste("Distribuição dos Grupos", tem_grupo_a_r and tem_grupo_b_r,
               f"(Centros: {centros_r})")
reportar_teste("Qualidade do Raio", raio_r < 10.0, f"(Raio Teórico: {raio_r:.4f})")


# =================================================================
# TESTE 3: CASO EXTREMO (k=N)
# =================================================================
print("\n>>> Testando Caso Extremo (k = N)...")
# Se k = numero de pontos, o raio deve ser 0.0 (cada ponto é seu próprio centro)
centros_extremo, raio_extremo = cluster_guloso(matriz_D, k=n_total)

reportar_teste("Raio Zero para k=N", raio_extremo == 0.0, f"(Raio: {raio_extremo})")
reportar_teste("Todos pontos viraram centros", len(set(centros_extremo)) == n_total)