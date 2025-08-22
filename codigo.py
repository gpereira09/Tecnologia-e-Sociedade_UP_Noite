# -*- coding: utf-8 -*-

import streamlit as st
import pandas as pd
import numpy as np
import requests
import json
import io
import time

# Título da página no Streamlit
st.set_page_config(layout="wide", page_title="Observatório de SST")

# Constantes e configurações
API_URL = "https://example.com/api/data"  # URL de exemplo. Substitua pela URL da API real (e.g., dados.gov.br)
DATA_FILE_PATH = "data/acidentes_sst.csv"
DATA_GENERATION_DELAY = 2 # Simula o tempo de carregamento da API.

# --- Funções de Ajuda ---

def exibir_loading():
    """Exibe um indicador de progresso enquanto a simulação de dados carrega."""
    with st.spinner("Carregando dados... Isso pode levar alguns segundos."):
        time.sleep(DATA_GENERATION_DELAY)

def pcm_to_wav(pcm_data, sample_rate):
    """
    Converte dados PCM brutos em um formato WAV para reprodução.
    Esta função é necessária para reproduzir áudio de APIs que retornam
    dados PCM. (Não usada neste projeto, mas mantida como exemplo)
    """
    header = np.array([
        0x52494646,  # 'RIFF'
        len(pcm_data) + 36,  # Tamanho do arquivo
        0x57415645,  # 'WAVE'
        0x666d7420,  # 'fmt '
        16,  # Tamanho do chunk
        1,  # Formato de áudio (PCM)
        1,  # Número de canais
        sample_rate,  # Taxa de amostragem
        sample_rate * 2,  # Taxa de bytes
        2,  # Alinhamento de bloco
        16,  # Bits por amostra
        0x64617461,  # 'data'
        len(pcm_data)  # Tamanho dos dados
    ], dtype='<u4')
    
    return io.BytesIO(header.tobytes() + pcm_data.tobytes())

# --- Simulação de Extração de Dados ---

def extrair_dados_api():
    """
    Simula a extração de dados de uma API governamental.
    
    Na implementação real do projeto, esta função deve fazer uma requisição
    HTTP para a URL da API (e.g., dados.gov.br) e retornar os dados.
    Aqui, criamos dados de exemplo para demonstração.
    """
    st.info("Simulando a extração de dados da API...")
    exibir_loading()

    # Criação de dados fictícios para demonstração
    num_entries = 1000
    estados = ["AC", "AL", "AM", "AP", "BA", "CE", "DF", "ES", "GO", "MA", "MG", "MS", "MT", "PA", "PB", "PE", "PI", "PR", "RJ", "RN", "RO", "RR", "RS", "SC", "SE", "SP", "TO"]
    data = {
        "id_acidente": range(1, num_entries + 1),
        "data": pd.to_datetime(pd.date_range("2024-01-01", periods=num_entries, freq="D")),
        "setor": np.random.choice(["Indústria", "Comércio", "Serviços", "Construção", "Agropecuária"], num_entries),
        "regiao": np.random.choice(["Sudeste", "Sul", "Nordeste", "Centro-Oeste", "Norte"], num_entries),
        "tipo_lesao": np.random.choice(["Corte", "Fratura", "Contusão", "Queimadura", "Outros"], num_entries, p=[0.4, 0.2, 0.2, 0.1, 0.1]),
        "origem": np.random.choice(["Equipamento", "Queda", "Manuseio", "Ambiental", "Trânsito"], num_entries, p=[0.3, 0.25, 0.25, 0.1, 0.1]),
        "estado": np.random.choice(estados, num_entries)
    }
    
    # Criar um DataFrame a partir dos dados fictícios
    df = pd.DataFrame(data)
    
    st.success("Dados extraídos com sucesso!")
    return df

# --- Tratamento de Dados com Pandas ---

def tratar_dados(df):
    """
    Lida com o tratamento dos dados, como limpeza e padronização.
    Esta função corresponde à tarefa 'Criar pipeline de limpeza de dados' do seu planejamento.
    """
    st.info("Iniciando o tratamento dos dados...")

    # Exemplo de tratamento de dados:
    # 1. Remover dados nulos (se existirem)
    df = df.dropna()
    
    # 2. Padronizar o nome das colunas
    df.columns = [col.lower().replace(" ", "_") for col in df.columns]
    
    # 3. Garantir que as colunas de data estão no formato correto
    df['data'] = pd.to_datetime(df['data'])
    
    # 4. Criar colunas adicionais para análise (ex: mês, ano)
    df['mes'] = df['data'].dt.month
    df['ano'] = df['data'].dt.year
    
    st.success("Dados tratados com sucesso!")
    return df

# --- Geração de KPIs e Dashboards com Streamlit ---

def gerar_dashboards(df):
    """
    Cria os dashboards e visualizações para o usuário final.
    Esta função utiliza o Streamlit para renderizar a interface.
    """
    st.header("Indicadores de Segurança no Trabalho")
    st.markdown("---")
    
    # Filtro para o Paraná
    df_parana = df[df['estado'] == 'PR']
    total_acidentes_parana = df_parana.shape[0]

    # KPI 1: Número Total de Acidentes (geral)
    total_acidentes = df.shape[0]
    st.subheader(f"Número Total de Acidentes (Geral): :blue[{total_acidentes}]")
    
    # KPI 2: Número de Acidentes no Paraná
    st.subheader(f"Acidentes Registrados no Paraná: :red[{total_acidentes_parana}]")
    st.markdown("---")

    # Gráfico de Acidentes por Setor no Paraná
    st.subheader("Acidentes por Setor no Paraná")
    acidentes_por_setor_pr = df_parana['setor'].value_counts()
    st.bar_chart(acidentes_por_setor_pr)
    
    st.markdown("---")
    
    # Gráfico de Acidentes por Tipo de Lesão no Paraná
    st.subheader("Acidentes por Tipo de Lesão no Paraná")
    acidentes_por_lesao_pr = df_parana['tipo_lesao'].value_counts()
    st.bar_chart(acidentes_por_lesao_pr)

    st.markdown("---")

    # Gráfico de Acidentes por Origem no Paraná
    st.subheader("Acidentes por Origem no Paraná")
    acidentes_por_origem_pr = df_parana['origem'].value_counts()
    st.bar_chart(acidentes_por_origem_pr)

    st.markdown("---")
    
    # Filtros e Tabela de Dados
    st.subheader("Dados Brutos e Filtros")
    
    setores_unicos = df['setor'].unique()
    regioes_unicas = df['regiao'].unique()
    tipos_lesao_unicos = df['tipo_lesao'].unique()
    origens_unicas = df['origem'].unique()

    setor_selecionado = st.selectbox("Selecione o Setor", ["Todos"] + list(setores_unicos))
    regiao_selecionada = st.selectbox("Selecione a Região", ["Todas"] + list(regioes_unicas))
    tipo_lesao_selecionada = st.selectbox("Selecione o Tipo de Lesão", ["Todos"] + list(tipos_lesao_unicos))
    origem_selecionada = st.selectbox("Selecione a Origem do Acidente", ["Todas"] + list(origens_unicas))

    df_filtrado = df.copy()
    if setor_selecionado != "Todos":
        df_filtrado = df_filtrado[df_filtrado['setor'] == setor_selecionado]
    if regiao_selecionada != "Todas":
        df_filtrado = df_filtrado[df_filtrado['regiao'] == regiao_selecionada]
    if tipo_lesao_selecionada != "Todos":
        df_filtrado = df_filtrado[df_filtrado['tipo_lesao'] == tipo_lesao_selecionada]
    if origem_selecionada != "Todas":
        df_filtrado = df_filtrado[df_filtrado['origem'] == origem_selecionada]
        
    st.write(f"Mostrando {df_filtrado.shape[0]} registros.")
    st.dataframe(df_filtrado)

# --- Função Principal ---

def main():
    """
    Função principal que orquestra a execução das etapas do projeto.
    """
    st.title("Sistema Agregador de Dados de SST")
    st.markdown("""
        Bem-vindo ao sistema de agregação e visualização de dados de segurança no trabalho.
        Este projeto extrai, trata e apresenta dados em um dashboard interativo.
    """)
    
    if st.button("Executar Projeto"):
        # Etapa 1: Extração
        df_bruto = extrair_dados_api()
        
        # Etapa 2: Tratamento
        df_tratado = tratar_dados(df_bruto)
        
        # Etapa 3: Visualização
        gerar_dashboards(df_tratado)

# Executa a função principal quando o script é iniciado
if __name__ == "__main__":
    main()

