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