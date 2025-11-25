import numpy as np
from distances import gerar_matriz_distancias

def reportar(teste_nome, obtido, esperado):
    # Função auxiliar para imprimir bonitinho
    erro = abs(obtido - esperado)
    status = "SUCESSO" if erro < 1e-5 else "FALHA"
    print(f"[{status}] {teste_nome}")
    print(f"   Esperado: {esperado}")
    print(f"   Obtido:   {obtido:.5f}")
    print("-" * 30)

print("=== INICIANDO BATERIA DE TESTES ===\n")

# ---------------------------------------------------------
# CASO 1: Teste Manual de Minkowski (Manhattan p=1)
# ---------------------------------------------------------
# Pontos: (0,0) e (3,4)
# Conta: |3-0| + |4-0| = 3 + 4 = 7
dados_teste_1 = np.array([
    [0, 0],
    [3, 4]
])
matriz = gerar_matriz_distancias(dados_teste_1, 'minkowski', p=1)
dist_obtida = matriz[0, 1]
reportar("Manhattan (3,4)", dist_obtida, 7.0)


# ---------------------------------------------------------
# CASO 2: Teste Manual de Minkowski (Euclidiana p=2)
# ---------------------------------------------------------
# Pontos: (0,0) e (3,4)
# Conta: sqrt(3^2 + 4^2) = sqrt(9 + 16) = sqrt(25) = 5
matriz = gerar_matriz_distancias(dados_teste_1, 'minkowski', p=2)
dist_obtida = matriz[0, 1]
reportar("Euclidiana (Triângulo 3-4-5)", dist_obtida, 5.0)


# ---------------------------------------------------------
# CASO 3: Teste de Simetria da Matriz
# ---------------------------------------------------------
# A distância de A->B tem que ser igual a B->A
dados_teste_2 = np.random.rand(10, 5) # 10 pontos aleatórios
matriz = gerar_matriz_distancias(dados_teste_2, 'minkowski', p=2)

simetria_ok = np.allclose(matriz, matriz.T)
diagonal_zerada = np.allclose(np.diag(matriz), 0)

print(f"[VERIFICAÇÃO] Matriz Simétrica? {simetria_ok}")
print(f"[VERIFICAÇÃO] Diagonal Zerada?  {diagonal_zerada}")
print("-" * 30)


# ---------------------------------------------------------
# CASO 4: O Teste de Fogo de Mahalanobis
# ---------------------------------------------------------
# TRUQUE MATEMÁTICO:
# Se os dados forem perfeitamente "esféricos" (variância 1 e covariância 0),
# a distância de Mahalanobis DEVE ser igual à distância Euclidiana.
# Vamos criar um dataset artificial que imita isso.

# Dataset: [1,0], [0,1], [-1,0], [0,-1] (Cruz em volta da origem)
# A covariância disso é próxima da Identidade (com ajuste de escala)
dados_cruz = np.array([
    [1.0, 0.0],
    [0.0, 1.0], 
    [-1.0, 0.0], 
    [0.0, -1.0]
])

# Distância Euclidiana entre (1,0) e (-1,0) é 2.0.
mat_euclid = gerar_matriz_distancias(dados_cruz, 'minkowski', p=2)
dist_euclid = mat_euclid[0, 2] # Distância entre ponto 0 e 2

# Distância Mahalanobis
# Como a covariância é diagonal, o Mahalanobis vai "normalizar" pelo desvio padrão.
# Neste dataset específico, a variância é 0.5 para cada eixo.
# S = [[0.5, 0], [0, 0.5]]. S_inv = [[2, 0], [0, 2]].
# Dist^2 = delta.T * S_inv * delta = [2, 0] * [[2,0],[0,2]] * [2,0]^T = 4 * 2 = 8
# Dist = sqrt(8) = 2.8284...
mat_maha = gerar_matriz_distancias(dados_cruz, 'mahalanobis')
dist_maha = mat_maha[0, 2]

# Verificando se a lógica bate (Maha deve ser sqrt(2) vezes maior que Euclid neste caso específico devido à variância 0.5)
fator_escala = np.sqrt(1 / 0.5) # sqrt(1 / variancia)
esperado_maha = dist_euclid * fator_escala # 2.0 * 1.414 = 2.828...

reportar("Mahalanobis (Consistência Estatística)", dist_maha, esperado_maha)