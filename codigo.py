# -*- coding: utf-8 -*-

import streamlit as st
import pandas as pd
import numpy as np
import io, csv, unicodedata, re
from typing import Optional, Tuple, List, Dict

st.set_page_config(page_title="Observatório — CSV (fix)", layout="wide")

# ---------------- Utilidades ----------------
def _strip_accents(s: str) -> str:
    if not isinstance(s, str):
        return s
    return ''.join(ch for ch in unicodedata.normalize('NFKD', s) if not unicodedata.combining(ch))

def normalize_name(c: str) -> str:
    c0 = _strip_accents(str(c).strip()).lower()
    c0 = re.sub(r'[^0-9a-z]+', '_', c0)
    c0 = re.sub(r'_+', '_', c0).strip('_')
    return c0

def normalize_headers(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [normalize_name(c) for c in df.columns]
    return df

def ensure_unique_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Remove colunas duplicadas mantendo a primeira ocorrência."""
    dup = df.columns.duplicated()
    return df.loc[:, ~dup].copy()

def sniff_delimiter(sample_bytes: bytes) -> Optional[str]:
    try:
        dialect = csv.Sniffer().sniff(sample_bytes.decode("latin1", errors="ignore"))
        return dialect.delimiter
    except Exception:
        return None

def _read_bytes(src) -> bytes:
    if hasattr(src, "read"):
        return src.read()
    elif isinstance(src, (bytes, bytearray)):
        return bytes(src)
    else:
        with open(src, "rb") as f:
            return f.read()

ENCODINGS_BR = ["latin1", "utf-8-sig", "utf-8", "cp1252"]

@st.cache_data(show_spinner=False)
def load_csv_simple(src,
                    sep_opt: Optional[str] = None,
                    decimal_opt: str = ",",
                    skiprows: int = 0,
                    encodings: List[str] = ENCODINGS_BR) -> Tuple[pd.DataFrame, str, str]:
    raw = _read_bytes(src)
    seps = [sep_opt] if sep_opt else []
    auto = sniff_delimiter(raw[:65536])
    if auto: seps.append(auto)
    for s in [";", ",", "\t", "|"]:
        if s not in seps:
            seps.append(s)
    last_err = None
    for enc in encodings:
        for sep in seps:
            try:
                bio = io.BytesIO(raw)
                df = pd.read_csv(
                    bio, sep=sep, engine="python", encoding=enc,
                    on_bad_lines="skip", quotechar='"', escapechar="\\",
                    skiprows=skiprows, decimal=decimal_opt
                )
                if df.shape[1] >= 1:
                    return df, enc, sep
            except Exception as e:
                last_err = e
                continue
    raise RuntimeError(f"Falha ao ler o CSV. Último erro: {last_err}")

def ensure_datetime(df: pd.DataFrame, col: Optional[str]) -> Optional[str]:
    if not col or col not in df.columns:
        return None
    try:
        df[col] = pd.to_datetime(df[col], errors="coerce", dayfirst=True)
    except Exception:
        df[col] = pd.to_datetime(df[col], errors="coerce")
    return col

# ---- fallback de gradient (se não houver matplotlib) ----
def style_heatmap(df: pd.DataFrame, cmap: str = "Blues"):
    try:
        import matplotlib  # noqa
        return df.style.background_gradient(cmap=cmap)
    except Exception:
        return df

# --------------- Mapeamento fixo p/ seu CSV ---------------
FIX_MAP = {
    "data":   "data_acidente",
    "uf":     "uf_munic_acidente",
    "setor":  "cnae2_0_empregador_1",     # descrição do CNAE
    "cnae_codigo": "cnae2_0_empregador",  # código numérico do CNAE
    "lesao":  "natureza_da_lesao",
    "origem": "agente_causador_acidente",
    "tipo_acidente": "tipo_do_acidente",
}

# --------------- UF/Região helpers ---------------
UF_SIGLAS = {
    "acre":"AC","alagoas":"AL","amapa":"AP","amazonas":"AM","bahia":"BA",
    "ceara":"CE","distrito federal":"DF","espirito santo":"ES","goias":"GO",
    "maranhao":"MA","mato grosso":"MT","mato grosso do sul":"MS",
    "minas gerais":"MG","para":"PA","paraiba":"PB","parana":"PR",
    "pernambuco":"PE","piaui":"PI","rio de janeiro":"RJ",
    "rio grande do norte":"RN","rio grande do sul":"RS",
    "rondonia":"RO","roraima":"RR","santa catarina":"SC",
    "sao paulo":"SP","sergipe":"SE","tocantins":"TO"
}
UF_REGIAO = {
    "AC":"Norte","AP":"Norte","AM":"Norte","PA":"Norte","RO":"Norte","RR":"Norte","TO":"Norte",
    "AL":"Nordeste","BA":"Nordeste","CE":"Nordeste","MA":"Nordeste","PB":"Nordeste","PE":"Nordeste","PI":"Nordeste","RN":"Nordeste","SE":"Nordeste",
    "DF":"Centro-Oeste","GO":"Centro-Oeste","MT":"Centro-Oeste","MS":"Centro-Oeste",
    "ES":"Sudeste","MG":"Sudeste","RJ":"Sudeste","SP":"Sudeste",
    "PR":"Sul","RS":"Sul","SC":"Sul"
}
def normalize_uf_name(x: str) -> str:
    x = _strip_accents(str(x)).strip().lower()
    x = re.sub(r'\s+', ' ', x)
    return x
def derive_sigla_from_name(x: str) -> Optional[str]:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return None
    s = str(x).strip()
    if re.fullmatch(r'[A-Za-z]{2}', s):
        return s.upper()
    key = normalize_uf_name(s)
    return UF_SIGLAS.get(key)
def derive_regiao_from_sigla(sigla: Optional[str]) -> Optional[str]:
    if not sigla:
        return None
    return UF_REGIAO.get(sigla.upper())

# --------------- UI / Fonte ---------------
st.title("Observatório — CSV (mapeado)")
with st.sidebar:
    st.header("Fonte do CSV")
    mode = st.radio("Como fornecer o arquivo?", ["Carregar arquivo", "Informar caminho local"], index=0)
    default_path = "D.SDA.PDA.005.CAT.202505.csv"
    upload = None
    path = None
    if mode == "Carregar arquivo":
        upload = st.file_uploader("Envie seu .csv", type=["csv"])
    else:
        path = st.text_input("Caminho do CSV", value=default_path)

    with st.expander("Opções avançadas (se precisar)"):
        sep_label = st.selectbox("Delimitador", ["Automático", ";", ",", "\\t", "|"], index=0)
        sep_opt = None if sep_label == "Automático" else ("\t" if sep_label == "\\t" else sep_label)
        decimal_opt = st.selectbox("Separador decimal", [",", "."], index=0)
        skiprows = st.number_input("Pular linhas iniciais", min_value=0, max_value=500, value=0, step=1)
        enc_first = st.selectbox("Encoding preferido", ["latin1 (BR)", "utf-8-sig", "utf-8", "cp1252"], index=0)
        enc_order = [enc_first.split(" ")[0]] + [e for e in ENCODINGS_BR if e != enc_first.split(" ")[0]]
    run = st.button("Carregar dados")

# --------------- Carregar & preparar ---------------
if not run:
    st.info("👈 Selecione/aponte o CSV and clique em **Carregar dados**.")
    st.stop()

src = upload if upload is not None else path
if src in (None, ""):
    st.error("Selecione o arquivo ou informe um caminho.")
    st.stop()

try:
    df_raw, enc_used, sep_used = load_csv_simple(src, sep_opt=sep_opt, decimal_opt=decimal_opt,
                                                 skiprows=skiprows, encodings=enc_order)
    st.success(f"CSV carregado. **Encoding:** {enc_used} | **Separador:** {repr(sep_used)} | Linhas: {df_raw.shape[0]:,} | Colunas: {df_raw.shape[1]}")
except Exception as e:
    st.error(f"Não consegui ler seu CSV: {e}")
    st.stop()

# normalização + remoção de duplicadas
df = normalize_headers(df_raw)
df = ensure_unique_columns(df)

# limpa espaços
for c in df.select_dtypes(include=['object']).columns:
    df[c] = df[c].astype(str).str.strip()

# checa mapeamento existe
missing = [FIX_MAP[k] for k in FIX_MAP if FIX_MAP[k] not in df.columns]
if missing:
    st.error(f"As colunas esperadas não foram encontradas após normalização: {missing}")
    st.stop()

# renomeia para nomes curtos e remove duplicadas novamente (evita choque com nomes já existentes)
df = df.rename(columns={
    FIX_MAP["data"]: "data",
    FIX_MAP["uf"]: "uf",
    FIX_MAP["setor"]: "setor",
    FIX_MAP["cnae_codigo"]: "cnae_codigo",
    FIX_MAP["lesao"]: "lesao",
    FIX_MAP["origem"]: "origem",
    FIX_MAP["tipo_acidente"]: "tipo_acidente",
})
df = ensure_unique_columns(df)

# datas derivadas
ensure_datetime(df, "data")
if "data" in df:
    df["ano"] = df["data"].dt.year
    df["mes"] = df["data"].dt.to_period("M").astype(str)

# uf/região
df["uf_sigla"] = df["uf"].apply(derive_sigla_from_name)
df["regiao"] = df["uf_sigla"].apply(derive_regiao_from_sigla)

# --------------- Filtros globais ---------------
st.header("Filtros globais")
f1, f2, f3, f4, f5, f6 = st.columns(6)
df_f = df.copy()

# Filtro por UF
ufs = sorted([u for u in df["uf_sigla"].dropna().unique().tolist() if u])
uf_sel = f1.multiselect("UF (sigla)", ufs, default=["PR"] if "PR" in ufs else [])
if uf_sel:
    df_f = df_f[df_f["uf_sigla"].isin(uf_sel)]

# Filtro por Região
regioes = sorted([r for r in df["regiao"].dropna().unique().tolist() if r])
reg_sel = f2.multiselect("Região", regioes, default=[])
if reg_sel:
    df_f = df_f[df_f["regiao"].isin(reg_sel)]

# Filtro por Mês
if "mes" in df:
    meses = sorted(df["mes"].dropna().unique().tolist())
    mes_sel = f3.selectbox("Mês (YYYY-MM)", ["(todos)"] + meses, index=0)
    if mes_sel != "(todos)":
        df_f = df_f[df_f["mes"] == mes_sel]

# Filtro por Ano
if "ano" in df:
    anos = sorted(df["ano"].dropna().unique().tolist())
    ano_sel = f4.selectbox("Ano", ["(todos)"] + anos, index=0)
    if ano_sel != "(todos)":
        df_f = df_f[df_f["ano"] == ano_sel]

# Filtro por Tipo de Acidente
tipo_opts = sorted(df["tipo_acidente"].dropna().astype(str).unique().tolist())
tipo_sel = f5.multiselect("Tipo de acidente", tipo_opts, default=[])
if tipo_sel:
    df_f = df_f[df_f["tipo_acidente"].astype(str).isin(tipo_sel)]

# Filtro por CNAE (código)
cnae_codigos = sorted([c for c in df["cnae_codigo"].dropna().unique().tolist() if c])
cnae_sel = f6.multiselect("CNAE (código)", cnae_codigos, default=[])
if cnae_sel:
    df_f = df_f[df_f["cnae_codigo"].astype(str).isin(cnae_sel)]

# Filtro adicional por descrição do CNAE
cnae_descricoes = sorted([d for d in df["setor"].dropna().unique().tolist() if d])
cnae_desc_sel = st.multiselect("CNAE (setor/atividade)", cnae_descricoes, default=[], key="cnae_desc")
if cnae_desc_sel:
    df_f = df_f[df_f["setor"].astype(str).isin(cnae_desc_sel)]

with st.expander("🔎 Filtro por termo (texto livre)"):
    termo = st.text_input("Digite um termo para filtrar (procura em colunas de texto). Deixe vazio para ignorar.")
    if termo:
        termo_lower = termo.lower()
        text_cols = [c for c in df_f.columns if df_f[c].dtype == "object"]
        mask = pd.Series(False, index=df_f.index)
        for c in text_cols:
            mask = mask | df_f[c].astype(str).str.lower().str.contains(termo_lower, na=False)
        df_f = df_f[mask]

# --------------- Abas / Dashboards ---------------
tabs = st.tabs([
    "📊 Visão geral", "⏱ Série temporal", "🗺️ UF/Região",
    "🏭 Setor/CNAE", "🩹 Tipo de Lesão", "⚙️ Origem/Causa",
    "📋 Dados + Download"
])

# Visão geral
with tabs[0]:
    st.subheader("Visão geral (dados filtrados)")
    k1, k2, k3, k4 = st.columns(4)
    total = df_f.shape[0]
    with k1: st.metric("Registros", f"{total:,}")

    if "mes" in df_f and df_f["mes"].notna().any():
        serie = df_f.groupby("mes").size().sort_index()
        ultimo = int(serie.iloc[-1])
        delta = int(ultimo - (serie.iloc[-2] if len(serie) > 1 else 0))
        with k2: st.metric("Último mês (qtd.)", f"{ultimo:,}", delta=f"{delta:+,}")
    else:
        with k2: st.metric("Último mês (qtd.)", "—")

    with k3: st.metric("UFs cobertas", f"{df_f['uf_sigla'].nunique():,}")
    with k4: st.metric("Setores/Atividades", f"{df_f['setor'].nunique():,}")

    st.markdown("---")
    cA, cB = st.columns([2, 1])
    with cA:
        if "mes" in df_f and df_f["mes"].notna().any():
            st.caption("Registros por mês")
            st.line_chart(df_f.groupby("mes").size().sort_index())
        else:
            st.info("Não há coluna de mês derivada (verifique 'data_acidente').")
    with cB:
        top_n = st.number_input("Top N (rankings)", min_value=5, max_value=50, value=10, step=1)
        st.caption(f"Top {top_n} — UF")
        st.bar_chart(df_f["uf_sigla"].value_counts().head(top_n))
        st.caption(f"Top {top_n} — Setor/Atividade")
        st.bar_chart(df_f["setor"].astype(str).value_counts().head(top_n))

# Série temporal
with tabs[1]:
    st.subheader("Série temporal — registros por mês")
    if "mes" in df_f and df_f["mes"].notna().any():
        st.line_chart(df_f.groupby("mes").size().sort_index())
        st.markdown("###### Desagregar por dimensão (opcional)")
        dim = st.selectbox("Dimensão", ["(nenhuma)", "uf_sigla", "regiao", "setor", "lesao", "origem", "tipo_acidente"], index=0)
        if dim != "(nenhuma)":
            top5 = df_f[dim].astype(str).value_counts().head(5).index.tolist()
            st.caption("Top 5 categorias ao longo do tempo (por mês)")
            st.line_chart(
                df_f[df_f[dim].astype(str).isin(top5)]
                .groupby(["mes", dim])
                .size()
                .unstack(fill_value=0)
                .sort_index()
            )
    else:
        st.info("Não foi possível derivar 'mes' (verifique 'data_acidente').")

# UF/Região
with tabs[2]:
    st.subheader("Distribuição por UF e Região")
    colA, colB = st.columns(2)
    with colA:
        st.caption("Por UF (sigla)")
        st.bar_chart(df_f["uf_sigla"].value_counts())
    with colB:
        st.caption("Por Região")
        st.bar_chart(df_f["regiao"].value_counts())

    st.markdown("###### Cruzamento: UF × outra dimensão")
    dim = st.selectbox("Dimensão", ["(nenhuma)", "setor", "lesao", "origem", "tipo_acidente"], index=0, key="ufx")
    if dim != "(nenhuma)":
        piv = pd.pivot_table(df_f, index="uf_sigla", columns=dim, aggfunc="size", fill_value=0)
        st.dataframe(style_heatmap(piv, "Greens"), use_container_width=True)

# Setor/CNAE
with tabs[3]:
    st.subheader("Distribuição por Setor/Atividade Econômica")
    top_n = st.slider("Top N", 5, 50, 20, step=1, key="setorn")
    st.bar_chart(df_f["setor"].astype(str).value_counts().head(top_n))

    st.markdown("###### Cruzamento: Setor × UF")
    piv = pd.pivot_table(df_f, index="setor", columns="uf_sigla", aggfunc="size", fill_value=0)
    st.dataframe(style_heatmap(piv.head(40), "Blues"), use_container_width=True)

# Tipo de Lesão
with tabs[4]:
    st.subheader("Distribuição por Tipo de Lesão")
    top_n = st.slider("Top N", 5, 50, 20, step=1, key="lesaon")
    st.bar_chart(df_f["lesao"].astype(str).value_counts().head(top_n))

    cruzar = st.selectbox("Cruzar com:", ["(nenhuma)", "uf_sigla", "setor", "origem", "regiao", "tipo_acidente"], index=0)
    if cruzar != "(nenhuma)":
        piv = pd.pivot_table(df_f, index="lesao", columns=cruzar, aggfunc="size", fill_value=0)
        st.dataframe(style_heatmap(piv.head(40), "Oranges"), use_container_width=True)

# Origem/Causa
with tabs[5]:
    st.subheader("Distribuição por Origem/Causa (Agente Causador)")
    top_n = st.slider("Top N", 5, 50, 20, step=1, key="origemn")
    st.bar_chart(df_f["origem"].astype(str).value_counts().head(top_n))

    st.markdown("###### Cruzamento: Origem/Causa × UF")
    piv = pd.pivot_table(df_f, index="origem", columns="uf_sigla", aggfunc="size", fill_value=0)
    st.dataframe(style_heatmap(piv.head(40), "Purples"), use_container_width=True)

# Dados + Download
with tabs[6]:
    st.subheader("Dados brutos (após filtros)")
    st.write(f"Mostrando {df_f.shape[0]:,} registros.")
    st.dataframe(df_f, use_container_width=True)
    st.download_button("⬇️ Baixar CSV filtrado",
                       data=df_f.to_csv(index=False).encode("utf-8-sig"),
                       file_name="dados_filtrados.csv", mime="text/csv")

# Perfil opcional
with st.expander("🧭 Perfil do dataset"):
    n_rows, n_cols = df.shape
    st.caption(f"**Linhas:** {n_rows:,} | **Colunas:** {n_cols}")
    profile = pd.DataFrame({
        "coluna": df.columns,
        "dtype": [str(t) for t in df.dtypes.values],
        "n_nulos": [int(df[c].isna().sum()) for c in df.columns],
        "%_nulos": [round(df[c].isna().mean()*100, 2) for c in df.columns],
        "n_unicos": [int(df[c].nunique(dropna=True)) for c in df.columns],
    })
    st.dataframe(profile, use_container_width=True)