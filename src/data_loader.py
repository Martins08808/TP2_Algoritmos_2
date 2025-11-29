import numpy as np
import pandas as pd
import os

def carregar_dados_sinteticos(filepath):
    """
    Carrega um dataset a partir de um arquivo CSV.
    
    Suposição:
        - O arquivo é um CSV válido com cabeçalho.
        - As colunas de features são todas exceto a última.
        - A última coluna contém os labels verdadeiros (y).
    
    Args:
        filepath (str): Caminho completo para o arquivo .csv
        
    Returns:
        tuple: (X, y) onde X são os dados (numpy array) e y os labels (numpy array).
    """
    
    # Verificação de segurança simples
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"O arquivo não foi encontrado: {filepath}")

    print(f"   [Data Loader] Carregando dados de: {os.path.basename(filepath)}")
    
    # Carrega o arquivo usando Pandas
    # Se seus CSVs não tiverem cabeçalho (header), adicione header=None aqui.
    df = pd.read_csv(filepath)
    
    # Separação de Features (X) e Target (y)
    # iloc[:, :-1] pega todas as linhas e todas as colunas, exceto a última
    X = df.iloc[:, :-1].values
    
    # iloc[:, -1] pega a última coluna (target)
    y = df.iloc[:, -1].values
    
    return X, y

def carregar_dados_german():
    """
    Lê o CSV do diretório ../data/ (relativo a este script em src/),
    separa features e target.
    NOTA: NORMALIZAÇÃO REMOVIDA (Usa dados brutos).
    """
    # 1. Construção dinâmica do caminho
    # Pega o diretório onde ESTE arquivo .py está (ex: .../projeto/src)
    diretorio_atual = os.path.dirname(os.path.abspath(__file__))
    
    # Sobe um nível e entra em 'dataset' (ex: .../projeto/dataset/arquivo.csv)
    caminho_arquivo = os.path.join(diretorio_atual, '..', 'data', 'SouthGermanCredit_LIMPO.csv')
    
    # Normaliza o caminho (remove .. redundantes) para evitar erros em alguns OS
    caminho_arquivo = os.path.normpath(caminho_arquivo)

    if not os.path.exists(caminho_arquivo):
        # Tenta dar uma dica de onde ele procurou para facilitar o debug
        raise FileNotFoundError(f"Arquivo não encontrado no caminho: {caminho_arquivo}")

    # 2. Ler o CSV
    df = pd.read_csv(caminho_arquivo)
    
    # 3. Separar X (features) e y (target)
    if 'credit_risk' in df.columns:
        y = df['credit_risk'].values
        X_raw = df.drop(columns=['credit_risk']).values
    else:
        # Fallback caso o nome da coluna mude, assume que é a última
        y = df.iloc[:, -1].values
        X_raw = df.iloc[:, :-1].values

    # 4. Dados Brutos (Sem Normalização)
    X = X_raw.astype(float) 
    
    print(f"   -> Dados carregados de '{caminho_arquivo}'")
    print(f"   -> (SEM NORMALIZAÇÃO): {X.shape[0]} amostras, {X.shape[1]} atributos.")
    
    return X, y