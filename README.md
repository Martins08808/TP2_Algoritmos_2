# TP2: Algoritmos Aproximativos - K-Centros

Este projeto implementa algoritmos 2-aproximados (Guloso e Refinamento) para o problema dos k-centros, além de comparar métricas de distância (Minkowski e Mahalanobis).

## Pré-requisitos

- Python 3 ou superior (pode funcionar com versões anteriores)
- Pip (gerenciador de pacotes)

## Instalação

Recomenda-se o uso de um ambiente virtual para não conflitar com outras instalações do seu sistema.

1. **Crie o ambiente virtual (opcional, mas recomendado):**

   ```bash
   # Linux/Mac
   python3 -m venv venv
   source venv/bin/activate

   # Windows
   python -m venv venv
   .\venv\Scripts\activate
   ```

2. **Instale a dependências:**

   ```bash
    pip install -r requirements.txt
   ```

## Organização do Repositório

```bash
├── README.md
├── data
│   ├── Bankruptcy Prediction_LIMPO.csv
│   ├── Gesture Segmentation_LIMPO.csv
│   ├── HCV-Egy-Data_LIMPO.csv
│   ├── Hand Postures_LIMPO.csv
│   ├── Iranian Customer Churn_LIMPO.csv
│   ├── SouthGermanCredit_LIMPO.csv
│   ├── circles_LIMPO.csv
│   ├── online_shoppers_converted.csv
│   ├── winequality-red_LIMPO.csv
│   └── winequality-white_LIMPO.csv
├── requirements.txt
├── resultados
│   └── "resultados das execuções dos algoritmos
│        para cada dataset"
├── src
│   ├── algorithms.py
│   ├── data_generator.py
│   ├── data_loader.py
│   ├── distances.py
│   ├── experiment_runner.py
│   └── resultados.py
├── synthetic
│   └── "datasets sintéticos gerados"
├── Tabela_Resumo_Real.xlsx
├── Tabela_Resumo_Sintéticos.xlsx
└── relatorio.pdf
```

### data

Contém os conjuntos de dados do mundo real utilizados nos experimentos. Todos os arquivos possuem o sufixo `_LIMPO`, indicando que passaram por uma etapa de pré-processamento e limpeza.

- Exemplos: `Bankruptcy Prediction`, `SouthGermanCredit`, `winequality`, etc.

### synthetic

Diretório destinado ao armazenamento dos datasets artificiais gerados pelo script `data_generator.py`, utilizados para testes controlados de validação dos algoritmos.

### resultados

Pasta onde são salvos automaticamente os outputs das execuções, incluindo logs, métricas e visualizações geradas para cada dataset.

### src (Código Fonte)

Contém todo o código fonte do projeto:

- **Core:**

  - `algorithms.py`: Implementação das classes/funções dos algoritmos principais (ex: K-Center, Greedy, Refinement).
  - `distances.py`: Biblioteca de funções para cálculo de métricas de dissimilaridade (ex: Euclidiana, Manhattan, Minkowski).
  - `data_loader.py`: Módulo responsável pela leitura, parsing e carregamento dos arquivos CSV.
  - `data_generator.py`: Script para criação de dados sintéticos com parâmetros controlados (clusters, dimensões, ruído).

- **Execução e Testes:**
  - `experiment_runner.py`: Arquivo principal para orquestrar a execução completa dos experimentos.
  - `resultados.py`: Módulo responsável por compilar os dados brutos das execuções em uma tabela única.

### Raiz

- `requirements.txt`: Lista de dependências Python necessárias para rodar o projeto.
- `Tabela_Resumo_Real.xlsx`: Planilha consolidada com a comparação de desempenho dos algoritmos nos dados reais.
- `Tabela_Resumo_Sintéticos.xlsx`: Planilha consolidada com a comparação de desempenho nos dados sintéticos.
- `relatorio.pdf`: Relatório final com a documentação das decisões de implementação dos algoritmos, detalhamento dos experimentos e análise dos resultados.

## Como Executar

Certifique-se de estar na raiz do projeto e de que seu ambiente Python tenha as dependências instaladas.

### Geração dos Dados Sintéticos

Execute o script gerador para criar os datasets artificiais. Os arquivos serão salvos automaticamente na pasta synthetic/.

```bash
python3 src/data_generator.py
```

### Execução dos Experimentos

Para rodar a bateria de testes nos datasets (reais e sintéticos), utilize o comando abaixo. As métricas serão exportadas para o diretório resultados/.

Utilize as variáveis globais no topo do arquivo para selecionar quais datasets serão utilizados na execução.

```bash
python3 src/experiment_runner.py
```
