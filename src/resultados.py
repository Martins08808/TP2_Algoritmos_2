import pandas as pd
import glob
import os

# Configuração
PASTA_ENTRADA = 'resultados'
ARQUIVO_SAIDA = 'Tabela_Resumo_Final.xlsx'

def main():
    # 1. Ler todos os arquivos individuais
    arquivos = glob.glob(os.path.join(PASTA_ENTRADA, "*.csv"))
    print(f"Lendo {len(arquivos)} arquivos de resultados em '{PASTA_ENTRADA}'...")
    
    lista_dfs = []
    for arq in arquivos:
        try:
            # Tenta ler o CSV. Se der erro de separador, tente sep=';' dependendo do seu sistema
            df = pd.read_csv(arq)
            lista_dfs.append(df)
        except Exception as e:
            print(f"Erro ao ler {arq}: {e}")
    
    if not lista_dfs:
        print("Nenhum arquivo encontrado! Verifique o caminho ou se rodou o experimento.")
        return

    # 2. Juntar tudo num DataFrame gigante
    df_completo = pd.concat(lista_dfs, ignore_index=True)
    
    # --- LIMPEZA E PADRONIZAÇÃO ---
    
    # Preenche 'Parametro_Largura' com 'N/A' caso esteja vazio ou NaN
    # Isso é importante para algoritmos que não usam esse parametro não serem excluídos do agrupamento
    if 'Parametro_Largura' in df_completo.columns:
        df_completo['Parametro_Largura'] = df_completo['Parametro_Largura'].fillna('N/A')
    
    # Opcional: Remover espaços em branco dos nomes das colunas (ex: " Tempo" -> "Tempo")
    df_completo.columns = df_completo.columns.str.strip()

    # 3. Calcular Média e Desvio Padrão
    
    # Colunas que identificam o "cenário" do teste (excluindo Repeticao)
    # Estas são as colunas que ficarão fixas à esquerda na tabela final
    cols_agrupamento_alvo = [
        'Dataset', 
        'Algoritmo', 
        'Metrica_Dist', 
        'Parametro_Largura'
    ]
    
    # Filtra para usar apenas as que realmente existem no CSV lido
    colunas_agrupamento = [c for c in cols_agrupamento_alvo if c in df_completo.columns]
    
    # Colunas numéricas que queremos tirar a média (Métricas de desempenho)
    # Note que 'Repeticao' NÃO entra aqui, pois ela desaparece ao tirarmos a média
    metricas_alvo = ['Raio', 'Silhouette', 'ARI', 'Tempo']
    metricas = [m for m in metricas_alvo if m in df_completo.columns]
    
    if not colunas_agrupamento or not metricas:
        print("ERRO: As colunas do CSV não batem com o formato esperado.")
        print(f"Colunas encontradas: {list(df_completo.columns)}")
        return

    print(f"Agrupando por: {colunas_agrupamento}")
    print(f"Calculando estatísticas (Média/Std) para: {metricas}")
    
    # A mágica acontece aqui: Agrupa pelo cenário e calcula média/desvio das métricas
    tabela_resumo = df_completo.groupby(colunas_agrupamento)[metricas].agg(['mean', 'std'])
    
    # 4. Formatação e Salvamento
    tabela_resumo = tabela_resumo.round(4)
    
    # Salva
    try:
        tabela_resumo.to_excel(ARQUIVO_SAIDA)
        print(f"\nSUCESSO! Tabela gerada em: {ARQUIVO_SAIDA}")
        print("A coluna 'Repeticao' foi utilizada para calcular a média e depois removida.")
    except Exception as e:
        print(f"Erro ao salvar Excel (o arquivo está aberto?): {e}")

if __name__ == "__main__":
    main()