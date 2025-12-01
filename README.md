# TP2: Algoritmos Aproximativos - K-Centros

Este projeto implementa algoritmos 2-aproximados (Guloso e Refinamento) para o problema dos k-centros, além de comparar métricas de distância (Minkowski e Mahalanobis).

## Pré-requisitos

- Python 3.8 ou superior
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
│   ├── main.py
│   ├── resultados.py
│   ├── runner_south_german.py
│   ├── teste_validacao.py
│   └── testekcentro.py
└── synthetic
   └── "dasets sintéticos gerados"
```
