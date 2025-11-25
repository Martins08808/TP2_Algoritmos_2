import numpy as np

def distancia_minkowski(ponto_a, ponto_b, p):

    diferenca = np.abs(ponto_a - ponto_b)
    soma_potencias = (diferenca ** p).sum()
    return soma_potencias ** (1.0 / p)

def distancia_mahalanobis(ponto_a, ponto_b, matriz_inv_cov):
    
    #Calcula a distância de Mahalanobis usando S^-1 (inversa da matriz de covariância).
    
    delta = ponto_a - ponto_b
    # Operação: delta.T @ S^-1 @ delta
    dist_quadrada = delta @ matriz_inv_cov @ delta
    return np.sqrt(max(0.0, dist_quadrada))

def gerar_matriz_distancias(dados, nome_metrica, p=None):
    """    
    Parâmetros:
      - dados: numpy array (N x Features)
      - nome_metrica: string ('minkowski' ou 'mahalanobis')
      - p: Obrigatório se a métrica for 'minkowski' (1 ou 2).
                 Ignorado se for 'mahalanobis'.
    """
    n_registros = dados.shape[0]
    matriz_D = np.zeros((n_registros, n_registros))
    
    # Preparação Específica da Métrica
    S_inv = None
    
    if nome_metrica == 'mahalanobis':
        # Cálculo da inversa da covariância 
        cov = np.cov(dados, rowvar=False)
        epsilon = 1e-8 
        S_inv = np.linalg.inv(cov + np.eye(cov.shape[0]) * epsilon)
        
    elif nome_metrica == 'minkowski':
        # Validação de segurança
        if p is None:
            raise ValueError("Para Minkowski, você DEVE fornecer o 'p' (ex: 1 ou 2).")

    # Loop de Construção da Matriz
    for i in range(n_registros):
        for j in range(i + 1, n_registros):
            
            valor = 0.0
            
            if nome_metrica == 'minkowski':
                valor = distancia_minkowski(dados[i], dados[j], p)
                
            elif nome_metrica == 'mahalanobis':
                valor = distancia_mahalanobis(dados[i], dados[j], S_inv)
            
            # Preenchimento simétrico
            matriz_D[i, j] = valor
            matriz_D[j, i] = valor
            
    return matriz_D