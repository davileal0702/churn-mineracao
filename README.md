# Mineração de Dados na prática: churn em telecom

App Streamlit do tema **#1 — Análise de Dados Preditiva I** (processo de mineração de dados, métodos e algoritmos), disciplina Data Analytics e Business Intelligence — UNISO.

Demonstração para acompanhar os slides, usando a base pública **Telco Customer Churn** (IBM, 7.043 clientes). Cada página responde uma pergunta, na ordem da apresentação:

| Página | Pergunta | O que mostra |
|---|---|---|
| 1 · O caso | Dá para prever quem vai sair? | O problema, os números principais e o objetivo |
| 2 · O processo | Como um projeto de mineração acontece? | As 6 fases do CRISP-DM, clicáveis, cada uma com um visual do caso |
| 3 · Os métodos | Que perguntas a mineração responde? | Classificação, agrupamento (k-means) e associação (Apriori) |
| 4 · Os algoritmos | Como cada algoritmo decide e qual é melhor? | Árvore de Decisão, Regressão Logística, k-NN, Naive Bayes e Random Forest, com ajustes e comparação |
| 5 · Na prática | Como a empresa usa o modelo? | Simulador de risco de um cliente e lista de retenção priorizada |

## Antes de publicar

Abra `app.py` e troque `NOME_GRUPO = "Grupo X"` (linha ~40) pelo número do grupo.

## Rodar no seu computador

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Publicar e gerar o link (Streamlit Community Cloud, gratuito)

1. Crie um repositório **público** no GitHub (ex.: `churn-mineracao`).
2. Envie os arquivos mantendo esta estrutura:
   ```
   app.py
   requirements.txt
   README.md
   .streamlit/config.toml
   data/telco_churn.csv
   ```
   Pelo site do GitHub: *Add file → Upload files* e arraste a pasta inteira.
3. Acesse **share.streamlit.io**, entre com a conta do GitHub e clique em **Create app**.
4. Escolha o repositório, a branch `main` e o arquivo `app.py`.
5. Em **Advanced settings**, selecione **Python 3.12** ou **3.13**.
6. Clique em **Deploy**. Em alguns minutos o app fica no ar com um link `https://<nome>.streamlit.app`.

Dicas:
- Abra o link alguns minutos antes da apresentação: apps gratuitos “dormem” depois de um tempo sem acesso e levam ~1 minuto para acordar.
- A primeira abertura demora alguns segundos (carrega as bibliotecas e treina os modelos); depois fica em cache.

## Fonte dos dados

IBM — Telco Customer Churn: https://github.com/IBM/telco-customer-churn-on-icp4d
