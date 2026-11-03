"""
Mineração de Dados na prática: previsão de cancelamento (churn) em uma operadora de telecom
Data Analytics e Business Intelligence (UNISO)
Tema #1: Análise de Dados Preditiva I (processo, métodos e algoritmos)

Demonstração para acompanhar os slides. A teoria fica nos slides; aqui o caso aparece funcionando.
Para rodar:  streamlit run app.py
"""

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from mlxtend.frequent_patterns import apriori, association_rules
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_score,
    recall_score,
    roc_auc_score,
    silhouette_score,
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

# ---------------------------------------------------------------------------
# Configurações (troque o nome do grupo aqui)
# ---------------------------------------------------------------------------
NOME_GRUPO = "Grupo 1"
SEMENTE = 42
AZUL, VERMELHO, CINZA = "#2E86AB", "#E4572E", "#9AA5B1"
COR_CANCELOU = {"Sim": VERMELHO, "Não": AZUL}

st.set_page_config(page_title="Mineração de Dados: Churn", page_icon="📡", layout="wide")

DATA_PATH = Path(__file__).parent / "data" / "telco_churn.csv"

# ---------------------------------------------------------------------------
# Tradução da base para português
# ---------------------------------------------------------------------------
COLUNAS = {
    "customerID": "ID", "gender": "Gênero", "SeniorCitizen": "Idoso", "Partner": "Parceiro(a)",
    "Dependents": "Dependentes", "tenure": "Meses como cliente", "PhoneService": "Telefone",
    "MultipleLines": "Múltiplas linhas", "InternetService": "Internet",
    "OnlineSecurity": "Segurança online", "OnlineBackup": "Backup online",
    "DeviceProtection": "Proteção de aparelho", "TechSupport": "Suporte técnico",
    "StreamingTV": "Streaming de TV", "StreamingMovies": "Streaming de filmes",
    "Contract": "Contrato", "PaperlessBilling": "Fatura digital", "PaymentMethod": "Pagamento",
    "MonthlyCharges": "Mensalidade (US$)", "TotalCharges": "Total pago (US$)", "Churn": "Cancelou",
}
VALORES = {
    "Yes": "Sim", "No": "Não", "Female": "Feminino", "Male": "Masculino",
    "No internet service": "Sem internet", "No phone service": "Sem telefone",
    "Month-to-month": "Mensal", "One year": "1 ano", "Two year": "2 anos",
    "Electronic check": "Cheque eletrônico", "Mailed check": "Boleto pelo correio",
    "Bank transfer (automatic)": "Débito automático", "Credit card (automatic)": "Cartão de crédito",
}
NUMERICAS = ["Meses como cliente", "Mensalidade (US$)", "Total pago (US$)"]
SERV_INTERNET = ["Segurança online", "Backup online", "Proteção de aparelho", "Suporte técnico",
                 "Streaming de TV", "Streaming de filmes"]


def pct(v, casas=0):
    return f"{v:.{casas}%}".replace(".", ",")


def num(v, casas=0):
    return f"{v:,.{casas}f}".replace(",", "X").replace(".", ",").replace("X", ".")


@st.cache_data
def carregar():
    bruto = pd.read_csv(DATA_PATH)
    brancos = int((bruto["TotalCharges"].astype(str).str.strip() == "").sum())
    df = bruto.rename(columns=COLUNAS)
    df["Idoso"] = df["Idoso"].map({0: "Não", 1: "Sim"})
    df["Internet"] = df["Internet"].map({"DSL": "DSL", "Fiber optic": "Fibra óptica", "No": "Sem internet"})
    for c in df.columns:
        if c not in NUMERICAS and c not in ("ID", "Internet", "Idoso"):
            df[c] = df[c].map(lambda v: VALORES.get(v, v))
    df["Total pago (US$)"] = pd.to_numeric(df["Total pago (US$)"], errors="coerce").fillna(0.0)
    return bruto, df, brancos


@st.cache_data
def preparar(df):
    X = df.drop(columns=["ID", "Cancelou"])
    y = (df["Cancelou"] == "Sim").astype(int)
    Xc = pd.get_dummies(X, drop_first=True, prefix_sep=" = ", dtype=int)
    X_tr, X_te, y_tr, y_te = train_test_split(Xc, y, test_size=0.3, stratify=y, random_state=SEMENTE)
    return X, Xc, y, X_tr, X_te, y_tr, y_te


def dados():
    _, df, _ = carregar()
    return df, *preparar(df)


# ---------------------------------------------------------------------------
# Algoritmos
# ---------------------------------------------------------------------------
ALGORITMOS = {
    "Árvore de Decisão": "Faz uma sequência de perguntas sobre o cliente (“o contrato é mensal?”, "
                         "“tem fibra óptica?”) até chegar a uma resposta.",
    "Regressão Logística": "Dá um peso para cada característica; somando os pesos, chega a uma "
                           "probabilidade entre 0% e 100%.",
    "k-NN": "Procura os k clientes mais parecidos no histórico e segue o que a maioria deles fez.",
    "Naive Bayes": "Usa o Teorema de Bayes: compara quão comuns são as características do cliente "
                   "entre quem cancelou e quem ficou.",
    "Random Forest": "Treina centenas de árvores de decisão diferentes e faz uma votação entre elas.",
}
PADRAO = {
    "Árvore de Decisão": {"profundidade": 4, "balancear": False},
    "Regressão Logística": {"balancear": False},
    "k-NN": {"k": 25},
    "Naive Bayes": {},
    "Random Forest": {"arvores": 300, "balancear": False},
}


def construir(nome, p):
    pesos = "balanced" if p.get("balancear") else None
    if nome == "Árvore de Decisão":
        return DecisionTreeClassifier(max_depth=p["profundidade"], min_samples_leaf=20,
                                      class_weight=pesos, random_state=SEMENTE)
    if nome == "Regressão Logística":
        return Pipeline([("escala", StandardScaler()),
                         ("modelo", LogisticRegression(max_iter=2000, class_weight=pesos))])
    if nome == "k-NN":
        return Pipeline([("escala", StandardScaler()), ("modelo", KNeighborsClassifier(n_neighbors=p["k"]))])
    if nome == "Naive Bayes":
        return GaussianNB()
    return RandomForestClassifier(n_estimators=p["arvores"], max_depth=8, class_weight=pesos,
                                  random_state=SEMENTE, n_jobs=1)


@st.cache_resource(show_spinner="Treinando o modelo...")
def treinar(nome, params: tuple):
    _, _, _, _, X_tr, _, y_tr, _ = dados()
    return construir(nome, dict(params)).fit(X_tr, y_tr)


def modelo_padrao(nome):
    return treinar(nome, tuple(sorted(PADRAO[nome].items())))


def metricas(y, proba, limiar=0.5):
    pred = (proba >= limiar).astype(int)
    return {
        "pred": pred,
        "Acurácia": accuracy_score(y, pred),
        "Precisão": precision_score(y, pred, zero_division=0),
        "Recall": recall_score(y, pred, zero_division=0),
        "AUC": roc_auc_score(y, proba),
    }


@st.cache_data(show_spinner="Treinando os 5 algoritmos...")
def comparar_todos():
    _, _, _, _, _, X_te, _, y_te = dados()
    linhas, probas = [], {}
    for nome in ALGORITMOS:
        proba = modelo_padrao(nome).predict_proba(X_te)[:, 1]
        m = metricas(y_te, proba)
        linhas.append({"Algoritmo": nome, **{k: v for k, v in m.items() if k != "pred"}})
        probas[nome] = proba
    tabela = pd.DataFrame(linhas).sort_values("AUC", ascending=False).reset_index(drop=True)
    return tabela, probas


def melhor_algoritmo():
    return comparar_todos()[0].iloc[0]["Algoritmo"]


# ---------------------------------------------------------------------------
# Componentes visuais reutilizáveis
# ---------------------------------------------------------------------------
PAGINAS = ["1 · O caso", "2 · O processo", "3 · Os métodos", "4 · Os algoritmos", "5 · Na prática"]


def cabecalho(n, pergunta, conclusao):
    st.caption(f"PÁGINA {n} DE 5 · {PAGINAS[n - 1].split(' · ')[1].upper()}")
    st.title(pergunta)
    st.info(conclusao, icon="💡")


def ir_para(pagina):
    st.session_state["pagina"] = pagina


def botao_proxima(n):
    if n < len(PAGINAS):
        st.divider()
        st.button(f"Próxima: {PAGINAS[n].split(' · ')[1]} →", on_click=ir_para, args=(PAGINAS[n],),
                  type="primary")


def matriz_confusao(y, pred, altura=330):
    cm = confusion_matrix(y, pred)
    textos = [
        [f"<b>{cm[0, 0]}</b><br>ficaram e o modelo<br>disse que ficariam",
         f"<b>{cm[0, 1]}</b><br>ficaram, mas o modelo<br>deu alarme falso"],
        [f"<b>{cm[1, 0]}</b><br>cancelaram e o modelo<br><b>não</b> percebeu",
         f"<b>{cm[1, 1]}</b><br>cancelaram e o modelo<br>avisou a tempo"],
    ]
    cores = [[0.15, 0.6], [0.6, 0.15]]
    fig = go.Figure(go.Heatmap(
        z=cores, x=["Modelo disse: fica", "Modelo disse: cancela"], y=["Na real: ficou", "Na real: cancelou"],
        text=textos, texttemplate="%{text}", textfont={"size": 14}, showscale=False, hoverinfo="skip",
        colorscale=[[0, "#DCEFF7"], [1, "#FBE1DA"]], zmin=0, zmax=1,
    ))
    fig.update_layout(height=altura, margin=dict(l=10, r=10, t=10, b=10), yaxis_autorange="reversed")
    return fig, cm


def frase_resultado(cm):
    vn, fp, fn, vp = cm.ravel()
    return (f"De **{fn + vp}** clientes que cancelaram, o modelo avisou sobre **{vp}** "
            f"(**recall de {pct(vp / (vp + fn))}**). De cada 10 clientes que ele apontou, "
            f"**{num(10 * vp / max(vp + fp, 1), 1)}** realmente cancelaram (**precisão de {pct(vp / max(vp + fp, 1))}**).")


def taxa_por(df, coluna):
    t = df.groupby(coluna)["Cancelou"].apply(lambda s: (s == "Sim").mean()).rename("Taxa").reset_index()
    t["Clientes"] = df.groupby(coluna).size().values
    return t.sort_values("Taxa")


def grafico_taxa(df, coluna, altura=320):
    t = taxa_por(df, coluna)
    fig = px.bar(t, x="Taxa", y=coluna, orientation="h", text=t["Taxa"].map(pct),
                 color_discrete_sequence=[VERMELHO], hover_data={"Clientes": True, "Taxa": ":.1%"})
    fig.add_vline(x=(df["Cancelou"] == "Sim").mean(), line_dash="dash", line_color=CINZA,
                  annotation_text="média", annotation_position="top")
    fig.update_layout(height=altura, xaxis_tickformat=".0%", xaxis_title="% que cancelou", yaxis_title="",
                      margin=dict(l=10, r=10, t=30, b=10))
    return fig


# ---------------------------------------------------------------------------
# Página 1: O caso
# ---------------------------------------------------------------------------
def pagina_caso():
    df, X, *_ = dados()
    taxa = (df["Cancelou"] == "Sim").mean()
    perdida = df.loc[df["Cancelou"] == "Sim", "Mensalidade (US$)"].sum()
    cabecalho(1, "A operadora perde 1 em cada 4 clientes. Dá para prever quem vai sair?",
              "**Objetivo:** usar o histórico de clientes para estimar a chance de cada um cancelar "
              "e agir antes que ele vá embora.")

    c1, c2, c3 = st.columns(3)
    c1.metric("Clientes analisados", num(len(df)))
    c2.metric("Cancelaram no último mês", pct(taxa, 1))
    c3.metric("Receita mensal perdida", f"US$ {num(perdida)}")

    esq, dir_ = st.columns([1, 1])
    with esq:
        cont = df["Cancelou"].value_counts().rename_axis("Cancelou").reset_index(name="Clientes")
        fig = px.pie(cont, names="Cancelou", values="Clientes", hole=0.55, color="Cancelou",
                     color_discrete_map=COR_CANCELOU)
        fig.update_traces(texttemplate=[f"%{{label}}<br>{pct(v / cont['Clientes'].sum(), 1)}" for v in cont["Clientes"]],
                          textfont_size=15)
        fig.update_layout(height=330, showlegend=False, margin=dict(l=10, r=10, t=10, b=10),
                          annotations=[dict(text="Cancelou?", showarrow=False, font_size=16)])
        st.plotly_chart(fig)
    with dir_:
        st.markdown("#### O que temos")
        st.markdown(
            f"""
- **{num(len(df))} clientes** reais de uma operadora de telefone e internet
- **{X.shape[1]} características** de cada um: contrato, serviços, forma de pagamento,
  tempo de casa, mensalidade…
- Para cada cliente, sabemos se **cancelou ou não** no último mês

#### O que queremos
Uma lista dos clientes com **maior risco de cancelar**, para a equipe de retenção ligar
primeiro, já que não dá para oferecer desconto para todo mundo.
"""
        )
    with st.expander("Ver os dados"):
        st.dataframe(df.head(50), hide_index=True)
        st.caption("Fonte: base pública *Telco Customer Churn*, disponibilizada pela IBM.")
    botao_proxima(1)


# ---------------------------------------------------------------------------
# Página 2: O processo (CRISP-DM)
# ---------------------------------------------------------------------------
FASES = ["Negócio", "Dados", "Preparação", "Modelagem", "Avaliação", "Implantação"]


def bloco_fase(o_que, no_caso):
    st.markdown(f"**O que se faz:** {o_que}")
    st.markdown(f"**No nosso caso:** {no_caso}")


def pagina_processo():
    df, X, Xc, y, X_tr, X_te, y_tr, y_te = dados()
    cabecalho(2, "Como um projeto de mineração de dados acontece?",
              "Seguimos as 6 fases do **CRISP-DM**. Clique em cada fase para ver o que foi feito no caso.")

    fase = st.segmented_control("Fase", [f"{i}. {f}" for i, f in enumerate(FASES, 1)],
                                default="1. Negócio", key="fase", label_visibility="collapsed")
    fase = (fase or "1. Negócio").split(". ")[1]
    st.markdown(f"### {FASES.index(fase) + 1}. {fase}")

    if fase == "Negócio":
        bloco_fase("entender o problema da empresa e transformá-lo em uma pergunta que os dados respondam.",
                   "a operadora quer reduzir cancelamentos, mas só consegue fazer ofertas para parte dos clientes.")
        c1, c2, c3 = st.columns(3)
        with c1.container(border=True):
            st.markdown("**🏢 Problema de negócio**\n\nPerda de 26,5% dos clientes no mês.")
        with c2.container(border=True):
            st.markdown("**🎯 Objetivo de mineração**\n\nPrever a probabilidade de cada cliente cancelar.")
        with c3.container(border=True):
            st.markdown("**✅ Critério de sucesso**\n\nPegar a maioria de quem vai cancelar, com AUC acima de 0,80.")
        st.caption("Perder um cliente custa mais caro que uma oferta de retenção. Por isso, deixar escapar "
                   "quem vai cancelar é o erro mais grave.")

    elif fase == "Dados":
        bloco_fase("explorar os dados: o que tem, qual a qualidade e que padrões já aparecem.",
                   "olhamos a taxa de cancelamento em cada grupo de clientes.")
        opcoes = [c for c in X.columns]
        var = st.selectbox("Taxa de cancelamento por:", opcoes, index=opcoes.index("Contrato"))
        if var in NUMERICAS:
            fig = px.histogram(df, x=var, color="Cancelou", barmode="overlay", opacity=0.65, nbins=40,
                               color_discrete_map=COR_CANCELOU)
            fig.update_layout(height=340, yaxis_title="Clientes", margin=dict(l=10, r=10, t=10, b=10))
            st.plotly_chart(fig)
        else:
            st.plotly_chart(grafico_taxa(df, var))
        t = taxa_por(df, "Contrato").set_index("Contrato")["Taxa"]
        st.success(f"**Padrão encontrado:** com contrato **mensal**, {pct(t['Mensal'])} cancelam. "
                   f"Com contrato de **2 anos**, só {pct(t['2 anos'])}. Experimente também *Internet*, "
                   f"*Pagamento* e *Meses como cliente*.")

    elif fase == "Preparação":
        _, _, brancos = carregar()
        bloco_fase("limpar e transformar os dados para o formato que os algoritmos entendem. "
                   "Costuma ser a fase mais demorada.",
                   "corrigimos valores em branco, transformamos texto em números e separamos treino e teste.")
        c1, c2, c3 = st.columns(3)
        c1.metric("Valores em branco corrigidos", brancos, help="Clientes novos sem total pago: preenchido com 0.")
        c2.metric("Colunas de texto → números", f"{X.shape[1]} → {Xc.shape[1]}",
                  help="Cada categoria vira uma coluna 0/1 (one-hot encoding).")
        c3.metric("Treino / teste", f"{num(len(y_tr))} / {num(len(y_te))}",
                  help="70% para o modelo aprender; 30% guardados para testar com clientes que ele nunca viu.")
        cols_antes = ["Contrato", "Internet", "Meses como cliente"]
        cols_depois = [c for c in Xc.columns if c.startswith("Contrato =") or c.startswith("Internet =")]
        cols_depois.append("Meses como cliente")
        a, b = st.columns([2, 3])
        a.markdown("**Antes** (texto)")
        a.dataframe(df[cols_antes].head(6), hide_index=True)
        b.markdown("**Depois** (números 0/1)")
        b.dataframe(Xc[cols_depois].head(6), hide_index=True)

    elif fase == "Modelagem":
        bloco_fase("escolher algoritmos e treiná-los com os dados de treino.",
                   "treinamos 5 algoritmos de classificação com os mesmos dados.")
        tabela, _ = comparar_todos()
        t = tabela.sort_values("AUC")
        fig = px.bar(t, x="AUC", y="Algoritmo", orientation="h", text=t["AUC"].map(lambda v: num(v, 3)),
                     color_discrete_sequence=[AZUL])
        fig.update_layout(height=300, xaxis_range=[0.5, 0.9], yaxis_title="", xaxis_title="AUC (quanto maior, melhor)",
                          margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(fig)
        st.caption("Cada algoritmo é explorado em detalhe na página 4.")

    elif fase == "Avaliação":
        nome = melhor_algoritmo()
        bloco_fase("testar o modelo com dados que ele nunca viu e conferir se atende ao critério de sucesso.",
                   f"o melhor algoritmo foi a **{nome}**. Veja como ele se saiu nos {num(len(y_te))} clientes de teste.")
        _, probas = comparar_todos()
        m = metricas(y_te, probas[nome])
        a, b = st.columns([3, 2])
        fig, cm = matriz_confusao(y_te, m["pred"])
        a.plotly_chart(fig)
        with b:
            st.metric("AUC", num(m["AUC"], 3), help="Capacidade de separar quem cancela de quem fica. 0,5 = chute; 1 = perfeito.")
            st.markdown(frase_resultado(cm))
            st.success("Critério de sucesso atingido (AUC > 0,80)." if m["AUC"] > 0.8
                       else "Critério de sucesso ainda não atingido.")

    elif fase == "Implantação":
        nome = melhor_algoritmo()
        bloco_fase("colocar o modelo para gerar valor no dia a dia da empresa e acompanhar se ele continua funcionando.",
                   "o modelo ordena os clientes do maior para o menor risco, e a equipe de retenção liga primeiro para o topo da lista.")
        _, probas = comparar_todos()
        ordem = np.argsort(-probas[nome])
        capt = np.concatenate([[0], np.cumsum(y_te.values[ordem]) / y_te.sum()])
        frac = np.linspace(0, 1, len(capt))
        i20 = int(0.2 * len(y_te))
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=frac, y=capt, mode="lines", name="Usando o modelo", line=dict(color=AZUL, width=3)))
        fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Ligando ao acaso",
                                 line=dict(color=CINZA, dash="dash")))
        fig.add_trace(go.Scatter(x=[0.2], y=[capt[i20]], mode="markers+text", marker=dict(size=12, color=VERMELHO),
                                 text=[f"  {pct(capt[i20])} dos cancelamentos"], textposition="middle right",
                                 showlegend=False))
        fig.update_layout(height=360, xaxis_tickformat=".0%", yaxis_tickformat=".0%",
                          xaxis_title="% dos clientes contatados (do maior para o menor risco)",
                          yaxis_title="% dos cancelamentos alcançados", legend=dict(x=0.6, y=0.1),
                          margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(fig)
        st.success(f"Ligando para os **20% de clientes com maior risco**, a equipe alcança **{pct(capt[i20])} "
                   f"de todos os cancelamentos**. Ao acaso, alcançaria só 20%. A ferramenta completa está na página 5.")

    botao_proxima(2)


# ---------------------------------------------------------------------------
# Página 3: Os métodos
# ---------------------------------------------------------------------------
@st.cache_data
def tabela_itens(df):
    it = pd.DataFrame(index=df.index)
    for c in ["Telefone", "Múltiplas linhas", "Backup online", "Proteção de aparelho",
              "Streaming de TV", "Streaming de filmes", "Fatura digital", "Idoso"]:
        it[c] = df[c] == "Sim"
    it["Tem parceiro(a)"] = df["Parceiro(a)"] == "Sim"
    it["Tem dependentes"] = df["Dependentes"] == "Sim"
    it["Sem suporte técnico"] = df["Suporte técnico"] == "Não"
    it["Sem segurança online"] = df["Segurança online"] == "Não"
    for v in ["Mensal", "1 ano", "2 anos"]:
        it[f"Contrato {v}"] = df["Contrato"] == v
    for v in ["DSL", "Fibra óptica", "Sem internet"]:
        it[f"Internet {v}" if v != "Sem internet" else "Sem internet"] = df["Internet"] == v
    it["Paga com cheque eletrônico"] = df["Pagamento"] == "Cheque eletrônico"
    it["Cliente novo (até 12 meses)"] = df["Meses como cliente"] <= 12
    it["Cancelou"] = df["Cancelou"] == "Sim"
    return it


@st.cache_data
def regras_churn(itens, suporte, confianca):
    freq = apriori(itens, min_support=suporte, use_colnames=True, max_len=3)
    if freq.empty:
        return pd.DataFrame()
    r = association_rules(freq, num_itemsets=len(itens), metric="confidence", min_threshold=confianca)
    return r[r["consequents"] == frozenset({"Cancelou"})]


@st.cache_data
def agrupar(df, k):
    feats = ["Meses como cliente", "Mensalidade (US$)"]
    escala = StandardScaler().fit(df[feats])
    km = KMeans(n_clusters=k, n_init=10, random_state=SEMENTE).fit(escala.transform(df[feats]))
    centros = escala.inverse_transform(km.cluster_centers_)
    sil = silhouette_score(escala.transform(df[feats]), km.labels_, sample_size=2500, random_state=SEMENTE)
    return km.labels_, centros, sil


def pagina_metodos():
    df, X, Xc, y, X_tr, X_te, y_tr, y_te = dados()
    cabecalho(3, "Que perguntas a mineração de dados responde?",
              "Cada **método** responde uma pergunta diferente sobre os mesmos clientes.")

    cards = [
        ("🎯", "Classificação", "Preditivo · supervisionado", "Este cliente vai cancelar?"),
        ("🔵", "Agrupamento", "Descritivo · não supervisionado", "Que tipos de cliente existem?"),
        ("🔗", "Associação", "Descritivo · não supervisionado", "Que características aparecem juntas?"),
    ]
    for col, (icone, nome, tipo, pergunta) in zip(st.columns(3), cards):
        with col.container(border=True):
            st.markdown(f"### {icone} {nome}")
            st.caption(tipo)
            st.markdown(f"**“{pergunta}”**")

    aba1, aba2, aba3 = st.tabs(["🎯 Classificação", "🔵 Agrupamento", "🔗 Associação"])

    with aba1:
        nome = melhor_algoritmo()
        st.markdown("Aprende com clientes do passado, **dos quais já sabemos a resposta**, e prevê a "
                    "resposta para clientes novos.")
        _, probas = comparar_todos()
        dist = pd.DataFrame({"Probabilidade prevista": probas[nome],
                             "Na real": np.where(y_te.values == 1, "Cancelou", "Ficou")})
        fig = px.histogram(dist, x="Probabilidade prevista", color="Na real", barmode="overlay", nbins=30,
                           opacity=0.7, color_discrete_map={"Cancelou": VERMELHO, "Ficou": AZUL})
        fig.update_layout(height=340, xaxis_tickformat=".0%", yaxis_title="Clientes de teste",
                          margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(fig)
        med = dist.groupby("Na real")["Probabilidade prevista"].mean()
        st.success(f"Nos clientes de teste, quem **cancelou** recebeu em média **{pct(med['Cancelou'])}** de "
                   f"probabilidade; quem **ficou**, **{pct(med['Ficou'])}**. O modelo separa bem os dois grupos. "
                   f"(Modelo: {nome}. Os 5 algoritmos estão na página 4.)")

    with aba2:
        st.markdown("Junta clientes parecidos **sem saber quem cancelou**. Aqui com o algoritmo **k-means**, "
                    "usando tempo de casa e mensalidade. Só depois olhamos o cancelamento de cada grupo.")
        k = st.slider("Número de grupos (k)", 2, 8, 4)
        rotulos, centros, sil = agrupar(df, k)
        d = df[["Meses como cliente", "Mensalidade (US$)", "Cancelou", "Contrato"]].copy()
        d["g"] = rotulos
        perfil = d.groupby("g").agg(Clientes=("Cancelou", "size"),
                                    Meses=("Meses como cliente", "mean"),
                                    Mensalidade=("Mensalidade (US$)", "mean"),
                                    Taxa=("Cancelou", lambda s: (s == "Sim").mean()))
        mt, mm = d["Meses como cliente"].mean(), d["Mensalidade (US$)"].mean()
        nomes = {g: f"G{g + 1}: {'antigos' if r.Meses >= mt else 'novos'}, "
                    f"conta {'alta' if r.Mensalidade >= mm else 'baixa'}" for g, r in perfil.iterrows()}
        d["Grupo"] = d["g"].map(nomes)
        a, b = st.columns([3, 2])
        with a:
            fig = px.scatter(d, x="Meses como cliente", y="Mensalidade (US$)", color="Grupo", opacity=0.45,
                             category_orders={"Grupo": sorted(nomes.values())})
            fig.add_trace(go.Scatter(x=centros[:, 0], y=centros[:, 1], mode="markers", name="Centro do grupo",
                                     marker=dict(symbol="x", size=15, color="black", line=dict(width=2))))
            fig.update_layout(height=400, margin=dict(l=10, r=10, t=10, b=10))
            st.plotly_chart(fig)
        with b:
            perfil.index = [nomes[g] for g in perfil.index]
            perfil = perfil.sort_values("Taxa", ascending=False)
            vis = pd.DataFrame({
                "Clientes": perfil["Clientes"],
                "Meses (média)": perfil["Meses"].round(0).astype(int),
                "Mensalidade": perfil["Mensalidade"].map(lambda v: f"US$ {num(v)}"),
                "Cancelaram": perfil["Taxa"].map(pct),
            })
            st.dataframe(vis)
            st.caption(f"Silhueta = {num(sil, 2)} (de -1 a 1: quanto maior, mais separados os grupos).")
        pior = perfil.index[0]
        st.success(f"O grupo **{pior}** tem a maior taxa de cancelamento (**{pct(perfil['Taxa'].iloc[0])}**): "
                   "é o segmento prioritário para uma campanha de retenção.")

    with aba3:
        st.markdown("Procura combinações do tipo **SE {A e B} ENTÃO {C}**, como na análise de cesta de compras. "
                    "Aqui com o algoritmo **Apriori**, buscando perfis que levam ao cancelamento.")
        with st.expander("Ajustar limites"):
            c1, c2 = st.columns(2)
            sup = c1.slider("Suporte mínimo (quão comum o perfil é)", 0.02, 0.30, 0.05, 0.01)
            conf = c2.slider("Confiança mínima (% que cancelou)", 0.30, 0.90, 0.50, 0.05)
        itens = tabela_itens(df)
        regras = regras_churn(itens, sup, conf)
        if regras.empty:
            st.warning("Nenhuma regra com esses limites. Diminua o suporte ou a confiança.")
        else:
            regras = regras.sort_values("lift", ascending=False).head(10)
            vis = pd.DataFrame({
                "SE o cliente…": regras["antecedents"].apply(lambda s: " + ".join(sorted(s))),
                "ENTÃO": "cancela",
                "Clientes com esse perfil": (regras["antecedent support"] * len(df)).round().astype(int),
                "Desses, cancelaram (confiança)": regras["confidence"].map(pct),
                "Lift": regras["lift"].map(lambda v: f"{num(v, 1)}×"),
            })
            st.dataframe(vis, hide_index=True)
            top = regras.iloc[0]
            st.success(f"**SE {' + '.join(sorted(top['antecedents']))} ENTÃO cancela:** "
                       f"{pct(top['confidence'])} desses clientes cancelaram, "
                       f"**{num(top['lift'], 1)} vezes** a taxa média.")
            st.caption("Lift = quantas vezes o perfil aumenta a chance em relação à média. Lift 1 = sem relação.")
    botao_proxima(3)


# ---------------------------------------------------------------------------
# Página 4: Os algoritmos
# ---------------------------------------------------------------------------
def cor_risco(taxa):
    """Azul quando a maioria fica, vermelho quando a maioria cancela (mais forte = mais certeza)."""
    neutro, azul, verm = np.array([238, 241, 244]), np.array([126, 183, 211]), np.array([228, 87, 46])
    if taxa < 0.5:
        t, a, b = 1 - taxa / 0.5, neutro, azul
    else:
        t, a, b = min((taxa - 0.5) / 0.3, 1), neutro, verm
    r, g, bl = (a + (b - a) * t).astype(int)
    return f"#{r:02x}{g:02x}{bl:02x}"


def arvore_graphviz(modelo, colunas, prof_max):
    t = modelo.tree_
    total = t.weighted_n_node_samples[0]
    nos, arestas = [], []

    def pergunta(no):
        col, lim = colunas[t.feature[no]], t.threshold[no]
        if " = " in col:
            var, val = col.split(" = ", 1)
            return f"{var} = {val}?", "não", "sim"
        return f"{col} ≤ {num(lim, 1)}?", "sim", "não"

    def visitar(no, nivel):
        frac = t.weighted_n_node_samples[no] / total
        v = t.value[no][0]
        taxa = v[1] / v.sum()
        folha = t.children_left[no] == -1 or nivel == prof_max
        texto = f"{pct(frac)} dos clientes\n{pct(taxa)} cancelam"
        if folha:
            texto += "\n→ " + ("CANCELA" if taxa >= 0.5 else "FICA")
            forma = "box"
        else:
            q, esq, dir_ = pergunta(no)
            texto = f"{q}\n" + texto
            forma = "box"
        fonte = "white" if taxa >= 0.55 else "#1B2A35"
        nos.append(f'n{no} [label="{texto}", shape={forma}, fillcolor="{cor_risco(taxa)}", fontcolor="{fonte}", '
                   f'penwidth={2 if folha else 1}];')
        if not folha:
            for filho, rot in ((t.children_left[no], esq), (t.children_right[no], dir_)):
                arestas.append(f'n{no} -> n{filho} [label=" {rot}"];')
                visitar(filho, nivel + 1)

    visitar(0, 0)
    return ("digraph G { bgcolor=transparent; nodesep=0.25; ranksep=0.45; "
            'node [style="rounded,filled", fontname=Helvetica, fontsize=11, color="#7A8B99"]; '
            'edge [fontname=Helvetica, fontsize=10, color="#7A8B99"]; '
            + " ".join(nos) + " " + " ".join(arestas) + " }")


def como_decide(nome, modelo, params, df, Xc):
    if nome == "Árvore de Decisão":
        st.graphviz_chart(arvore_graphviz(modelo, list(Xc.columns), min(3, params["profundidade"])))
        st.caption("Cada caixa mostra quantos clientes chegaram até ali e quantos deles cancelaram. "
                   "Azul = a maioria fica; vermelho = a maioria cancela; quanto mais forte a cor, mais certeza. "
                   "Mostrando até 3 níveis de perguntas.")

    elif nome in ("Regressão Logística", "Random Forest"):
        if nome == "Regressão Logística":
            v = pd.Series(modelo.named_steps["modelo"].coef_[0], index=Xc.columns)
            titulo = "Peso de cada característica (vermelho aumenta a chance de cancelar, azul diminui)"
        else:
            v = pd.Series(modelo.feature_importances_, index=Xc.columns)
            titulo = "Características mais usadas pelas árvores para decidir"
        top = v.reindex(v.abs().sort_values(ascending=False).index).head(10)[::-1]
        fig = px.bar(x=top.values, y=top.index, orientation="h", color=top.values > 0,
                     color_discrete_map={True: VERMELHO, False: AZUL}, title=titulo)
        fig.update_layout(height=400, showlegend=False, xaxis_title="", yaxis_title="",
                          margin=dict(l=10, r=10, t=40, b=10))
        if nome == "Random Forest":
            fig.update_traces(marker_color=AZUL)
        st.plotly_chart(fig)

    elif nome == "k-NN":
        k = params["k"]
        c1, c2 = st.columns(2)
        meses = c1.slider("Cliente novo: meses como cliente", 0, 72, 6)
        mensal = c2.slider("Cliente novo: mensalidade (US$)", 18, 120, 90)
        feats = ["Meses como cliente", "Mensalidade (US$)"]
        amostra = df.sample(600, random_state=SEMENTE)
        esc = StandardScaler().fit(amostra[feats])
        Z = esc.transform(amostra[feats])
        z0 = esc.transform(pd.DataFrame([[meses, mensal]], columns=feats))
        idx = np.argsort(np.linalg.norm(Z - z0, axis=1))[:k]
        viz = amostra.iloc[idx]
        n_canc = int((viz["Cancelou"] == "Sim").sum())
        fig = px.scatter(amostra, x=feats[0], y=feats[1], color="Cancelou", opacity=0.25,
                         color_discrete_map=COR_CANCELOU)
        for _, r in viz.iterrows():
            fig.add_trace(go.Scatter(x=[meses, r[feats[0]]], y=[mensal, r[feats[1]]], mode="lines",
                                     line=dict(color=CINZA, width=1), showlegend=False, hoverinfo="skip"))
        fig.add_trace(go.Scatter(x=viz[feats[0]], y=viz[feats[1]], mode="markers", showlegend=False,
                                 marker=dict(size=10, color=[COR_CANCELOU[c] for c in viz["Cancelou"]],
                                             line=dict(width=1.5, color="black"))))
        fig.add_trace(go.Scatter(x=[meses], y=[mensal], mode="markers", name="Cliente novo",
                                 marker=dict(symbol="star", size=22, color="gold", line=dict(width=1.5, color="black"))))
        fig.update_layout(height=420, margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(fig)
        voto = "CANCELA" if n_canc > k / 2 else "FICA"
        st.markdown(f"Dos **{k} vizinhos** mais próximos da ⭐, **{n_canc} cancelaram** e {k - n_canc} ficaram "
                    f"→ previsão: **{voto}**.")
        st.caption("Ilustração com só 2 características e 600 clientes, para dar para desenhar. "
                   "O modelo de verdade usa todas as características.")

    elif nome == "Naive Bayes":
        pares = [("Contrato", "Mensal"), ("Internet", "Fibra óptica"), ("Pagamento", "Cheque eletrônico"),
                 ("Suporte técnico", "Não"), ("Dependentes", "Não")]
        linhas = []
        for col, val in pares:
            for grupo in ["Sim", "Não"]:
                sub = df[df["Cancelou"] == grupo]
                linhas.append({"Característica": f"{col}: {val}",
                               "Grupo": "Entre quem cancelou" if grupo == "Sim" else "Entre quem ficou",
                               "Frequência": (sub[col] == val).mean()})
        t = pd.DataFrame(linhas)
        fig = px.bar(t, x="Frequência", y="Característica", color="Grupo", barmode="group", orientation="h",
                     text=t["Frequência"].map(pct),
                     color_discrete_map={"Entre quem cancelou": VERMELHO, "Entre quem ficou": AZUL})
        fig.update_layout(height=400, xaxis_tickformat=".0%", yaxis_title="", margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(fig)
        st.caption("Exemplo: 89% de quem cancelou tinha contrato mensal, contra 43% de quem ficou. O Naive Bayes "
                   "multiplica essas comparações (supondo que as características são independentes) para "
                   "calcular a chance de cancelar.")


def pagina_algoritmos():
    df, X, Xc, y, X_tr, X_te, y_tr, y_te = dados()
    cabecalho(4, "Como cada algoritmo decide, e qual funciona melhor?",
              "Escolha um algoritmo, veja como ele toma a decisão e mexa nos ajustes para ver o resultado mudar.")

    nome = st.segmented_control("Algoritmo", list(ALGORITMOS), default="Árvore de Decisão", key="alg")
    nome = nome or "Árvore de Decisão"
    st.markdown(f"**{nome}:** {ALGORITMOS[nome]}")

    with st.container(border=True):
        st.markdown("**Ajustes**")
        p = dict(PADRAO[nome])
        c1, c2, c3 = st.columns(3)
        if nome == "Árvore de Decisão":
            p["profundidade"] = c1.slider("Profundidade (nº de perguntas seguidas)", 1, 15, 4)
        elif nome == "k-NN":
            p["k"] = c1.slider("k (nº de vizinhos consultados)", 1, 75, 25, step=2)
        elif nome == "Random Forest":
            p["arvores"] = c1.slider("Número de árvores", 10, 500, 300, step=10)
        else:
            c1.caption("Este algoritmo não tem ajuste principal relevante.")
        if "balancear" in p:
            p["balancear"] = c2.toggle("Dar mais peso a quem cancela", help="Faz o modelo se esforçar mais para "
                                       "não deixar escapar quem cancela (recall sobe, precisão cai).")
        limiar = c3.slider("Limiar: avisar a partir de que chance de cancelar", 5, 95, 50, 5, format="%d%%",
                           help="Abaixar o limiar faz o modelo avisar sobre mais clientes.") / 100

    modelo = treinar(nome, tuple(sorted(p.items())))

    st.markdown("#### Como ele decide")
    como_decide(nome, modelo, p, df, Xc)

    st.markdown("#### Resultado nos clientes de teste")
    m = metricas(y_te, modelo.predict_proba(X_te)[:, 1], limiar)
    c = st.columns(4)
    c[0].metric("Acurácia", pct(m["Acurácia"], 1), help="% de previsões certas no total.")
    c[1].metric("Precisão", pct(m["Precisão"], 1), help="Dos clientes que o modelo apontou, quantos cancelaram de fato.")
    c[2].metric("Recall", pct(m["Recall"], 1), help="De quem cancelou de fato, quantos o modelo pegou.")
    c[3].metric("AUC", num(m["AUC"], 3), help="Capacidade de separar quem cancela de quem fica. 0,5 = chute; 1 = perfeito.")
    a, b = st.columns([3, 2])
    fig, cm = matriz_confusao(y_te, m["pred"], altura=300)
    a.plotly_chart(fig)
    with b:
        st.markdown(frase_resultado(cm))
        acc_tr = accuracy_score(y_tr, (modelo.predict_proba(X_tr)[:, 1] >= limiar).astype(int))
        if acc_tr - m["Acurácia"] > 0.04:
            st.warning(f"Acerta {pct(acc_tr)} no treino, mas só {pct(m['Acurácia'])} no teste: o modelo "
                       "**decorou** os dados de treino (*overfitting*).")
        st.caption("Por que não olhar só a acurácia? 73% dos clientes ficam. Um “modelo” que sempre "
                   "diz “fica” acertaria 73% sem pegar nenhum cancelamento.")

    st.markdown("#### Comparação dos 5 algoritmos")
    tabela, _ = comparar_todos()
    longo = tabela.melt(id_vars="Algoritmo", value_vars=["Acurácia", "Precisão", "Recall", "AUC"],
                        var_name="Métrica", value_name="Valor")
    fig = px.bar(longo, x="Métrica", y="Valor", color="Algoritmo", barmode="group",
                 category_orders={"Algoritmo": list(tabela["Algoritmo"])},
                 color_discrete_sequence=[AZUL, "#7FB7D3", "#F2A541", "#5C6B77", "#B8C4CE"])
    fig.update_layout(height=340, yaxis_tickformat=".0%", yaxis_title="", xaxis_title="",
                      margin=dict(l=10, r=10, t=10, b=10))
    st.plotly_chart(fig)
    st.success(f"**{tabela.iloc[0]['Algoritmo']}** teve a melhor AUC ({num(tabela.iloc[0]['AUC'], 3)}), com os "
               "ajustes padrão e limiar de 50%. É o modelo usado na página 5.")
    botao_proxima(4)


# ---------------------------------------------------------------------------
# Página 5: Na prática
# ---------------------------------------------------------------------------
def pagina_pratica():
    df, X, Xc, y, X_tr, X_te, y_tr, y_te = dados()
    nome = melhor_algoritmo()
    modelo = modelo_padrao(nome)
    cabecalho(5, "Como a empresa usa o modelo no dia a dia?",
              "O atendimento consulta o risco de um cliente na hora, e a equipe de retenção recebe uma "
              f"lista de quem ligar primeiro. Modelo usado: **{nome}**.")

    aba1, aba2 = st.tabs(["🔮 Consultar um cliente", "📋 Lista para a equipe de retenção"])

    with aba1:
        ent, res = st.columns([3, 2])
        with ent:
            c1, c2 = st.columns(2)
            meses = c1.slider("Meses como cliente", 0, 72, 5)
            mensal = c2.slider("Mensalidade (US$)", 18, 120, 85)
            contrato = c1.selectbox("Contrato", ["Mensal", "1 ano", "2 anos"])
            internet = c2.selectbox("Internet", ["Fibra óptica", "DSL", "Sem internet"])
            formas = sorted(df["Pagamento"].unique())
            pagamento = c1.selectbox("Pagamento", formas, index=formas.index("Cheque eletrônico"))
            suporte = c2.radio("Suporte técnico", ["Não", "Sim"], horizontal=True)
            seguranca = c1.radio("Segurança online", ["Não", "Sim"], horizontal=True)
            idoso = c2.radio("Idoso (65+)", ["Não", "Sim"], horizontal=True)

        cliente = X.mode().iloc[0].copy()
        novos = {"Meses como cliente": float(meses), "Mensalidade (US$)": float(mensal),
                 "Total pago (US$)": float(meses * mensal), "Contrato": contrato, "Internet": internet,
                 "Pagamento": pagamento, "Suporte técnico": suporte, "Segurança online": seguranca,
                 "Idoso": idoso}
        for k, v in novos.items():
            cliente[k] = v
        for c in SERV_INTERNET:
            if internet == "Sem internet":
                cliente[c] = "Sem internet"
            elif cliente[c] == "Sem internet":
                cliente[c] = "Não"
        linha = pd.DataFrame([cliente])[X.columns]
        for c in NUMERICAS:
            linha[c] = linha[c].astype(float)
        linha = pd.get_dummies(linha, prefix_sep=" = ", dtype=int).reindex(columns=Xc.columns, fill_value=0)
        prob = float(modelo.predict_proba(linha)[0, 1])

        nivel, cor, acao = (
            ("ALTO", VERMELHO, "Ligar hoje: oferecer migração para contrato anual com desconto ou "
                               "suporte técnico grátis por 3 meses.") if prob >= 0.6 else
            ("MÉDIO", "#F2A541", "Incluir na campanha de relacionamento e acompanhar reclamações.") if prob >= 0.3 else
            ("BAIXO", "#3BB273", "Nenhuma ação especial."))
        with res:
            fig = go.Figure(go.Indicator(
                mode="gauge+number", value=prob * 100, number={"suffix": "%", "valueformat": ".0f"},
                title={"text": "Chance de cancelar"},
                gauge={"axis": {"range": [0, 100]}, "bar": {"color": cor},
                       "steps": [{"range": [0, 30], "color": "#E3F4E8"}, {"range": [30, 60], "color": "#FDF0DC"},
                                 {"range": [60, 100], "color": "#FBE1DA"}]}))
            fig.update_layout(height=260, margin=dict(l=35, r=35, t=50, b=0))
            st.plotly_chart(fig)
            st.markdown(f"### Risco {nivel}")
            st.markdown(f"**Ação sugerida:** {acao}")
        st.caption("Para demonstrar: mude o contrato de “Mensal” para “2 anos”, ou aumente os meses como cliente, "
                   "e veja a chance cair.")

    with aba2:
        proba = modelo.predict_proba(X_te)[:, 1]
        base = df.loc[X_te.index, ["ID", "Meses como cliente", "Contrato", "Internet", "Mensalidade (US$)"]].copy()
        base["Chance de cancelar"] = proba
        base["Cancelou de fato?"] = np.where(y_te.values == 1, "Sim", "Não")
        base = base.sort_values("Chance de cancelar", ascending=False)

        n = st.slider("Quantos clientes a equipe consegue ligar este mês?", 50, 1000, 300, 50)
        topo = base.head(n)
        acertos = int((topo["Cancelou de fato?"] == "Sim").sum())
        total = int((base["Cancelou de fato?"] == "Sim").sum())
        ao_acaso = n * total / len(base)
        c1, c2, c3 = st.columns(3)
        c1.metric("Ligações", n)
        c2.metric("Clientes que iam cancelar e foram alcançados", acertos)
        c3.metric("Ao acaso, seriam", num(ao_acaso), f"modelo {num(acertos / ao_acaso, 1)}× melhor",
                  delta_color="off")
        st.dataframe(topo.style.format({"Chance de cancelar": "{:.0%}", "Mensalidade (US$)": "{:.2f}"}),
                     hide_index=True, height=320)
        st.download_button("⬇️ Baixar lista (CSV)", topo.to_csv(index=False, sep=";", decimal=",").encode("utf-8-sig"),
                           "lista_retencao.csv", "text/csv")
        st.caption(f"Lista feita com os {num(len(base))} clientes de teste, que o modelo nunca viu. Por isso dá "
                   "para conferir na última coluna se quem estava no topo realmente cancelou.")


# ---------------------------------------------------------------------------
# Navegação
# ---------------------------------------------------------------------------
FUNCOES = dict(zip(PAGINAS, [pagina_caso, pagina_processo, pagina_metodos, pagina_algoritmos, pagina_pratica]))


def main():
    with st.sidebar:
        st.markdown("## 📡 Churn em Telecom")
        st.caption("Mineração de dados na prática")
        escolha = st.radio("Navegação", PAGINAS, key="pagina", label_visibility="collapsed")
        st.divider()
        st.caption(f"**{NOME_GRUPO}**  \nData Analytics e Business Intelligence (UNISO)  \n"
                   "Tema: Análise de Dados Preditiva I")
    FUNCOES[escolha]()


main()
