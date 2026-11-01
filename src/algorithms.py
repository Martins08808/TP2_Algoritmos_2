import numpy as np

# ALGORITMO GULOSO

def cluster_guloso(matriz_dist, k, semente=None):
    """
    Executa o algoritmo 2-aproximado guloso.
    
    Retorna:
      - centros: Lista de índices escolhidos.
      - raio_max: O raio final da solução (distância máxima de um ponto ao seu centro).
    """
    if semente is not None:
        np.random.seed(semente)
        
    n_pontos = matriz_dist.shape[0]
    
    # Escolha inicial arbitrária
    centros = [np.random.randint(0, n_pontos)]
    
    # Inicializa vetor com distâncias de cada ponto ao centro mais próximo
    dists_minimas = matriz_dist[centros[0], :].copy()
    
    # Loop para escolher os k-1 centros restantes
    for _ in range(1, k):
        # Seleciona o ponto mais distante dos centros já escolhidos
        candidato = np.argmax(dists_minimas)
        centros.append(candidato)
        
        # Atualiza distâncias mínimas: mantém a menor entre a atual e a do novo candidato
        dists_candidato = matriz_dist[candidato, :]
        dists_minimas = np.minimum(dists_minimas, dists_candidato)

    # O raio é a maior distância restante no vetor
    raio_max = np.max(dists_minimas)
    
    return centros, raio_max

# ALGORITMO COM REFINAMENTO DE INTERVALO (Busca Binária)
def _verificar_viabilidade(matriz_dist, raio_teste, k_max):
    """
    Verifica se é possível cobrir todos os pontos com 'raio_teste' usando <= k_max centros.
    Estratégia: Remover pontos cobertos (dist <= 2r) a cada iteração.
    """
    n_pontos = matriz_dist.shape[0]
    nao_cobertos = set(range(n_pontos))
    centros_escolhidos = []
    
    while nao_cobertos:
        # Seleciona um ponto arbitrário dos não cobertos
        pivo = nao_cobertos.pop()
        centros_escolhidos.append(pivo)
        
        if len(centros_escolhidos) > k_max:
            return False, []

        # Identifica pontos cobertos pelo pivô (expandido por 2r para garantir 2-aproximação)
        limite = 2 * raio_teste
        remover = [p for p in nao_cobertos if matriz_dist[pivo, p] <= limite]
        
        # Atualiza conjunto de não cobertos
        for p in remover:
            nao_cobertos.remove(p)
            
    return True, centros_escolhidos

def cluster_refinamento(matriz_dist, k, razao_parada=0.01):
    """
    Busca binária para encontrar o menor raio que satisfaz a cobertura.
    Parada: quando a largura do intervalo for < razao_parada * largura_inicial.
    """
    limite_inf = 0.0
    limite_sup = np.max(matriz_dist)
    
    meta = (limite_sup - limite_inf) * razao_parada
    
    melhores_centros = []
    melhor_raio_teorico = limite_sup
    
    # Busca Binária
    while (limite_sup - limite_inf) > meta:
        raio_meio = (limite_inf + limite_sup) / 2.0
        
        # Teste de viabilidade do raio
        viavel, centros_temp = _verificar_viabilidade(matriz_dist, raio_meio, k)
        
        if viavel:
            # Raio viável: tenta reduzir
            melhor_raio_teorico = raio_meio
            melhores_centros = centros_temp
            limite_sup = raio_meio
        else:
            # Raio inviável: precisa aumentar
            limite_inf = raio_meio
            
    # Preenchimento de segurança caso a busca retorne < k centros
    if len(melhores_centros) < k:
        disponiveis = list(set(range(matriz_dist.shape[0])) - set(melhores_centros))
        faltam = k - len(melhores_centros)
        
        if disponiveis:
            extras = np.random.choice(disponiveis, size=min(faltam, len(disponiveis)), replace=False)
            melhores_centros.extend(extras)

    return melhores_centros, melhor_raio_teorico