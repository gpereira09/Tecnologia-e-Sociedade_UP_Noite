    

# -*- coding: utf-8 -*-

import streamlit as st
import pandas as pd
import numpy as np
import io, csv, unicodedata, re
import os
import glob
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
        # Tenta detectar o dialeto do CSV
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
    
    # Tenta farejar o delimitador
    auto = sniff_delimiter(raw[:65536])
    if auto and auto not in seps: seps.append(auto)
    
    # Adiciona delimitadores comuns
    for s in [";", ",", "\t", "|"]:
        if s not in seps:
            seps.append(s)
    
    # Remove duplicados e None
    seps = [s for s in seps if s is not None]
    seps = list(dict.fromkeys(seps)) # Remove duplicados mantendo a ordem

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
        # Tenta formato brasileiro (dia primeiro)
        df[col] = pd.to_datetime(df[col], errors="coerce", dayfirst=True)
    except Exception:
        # Tenta formato padrão
        df[col] = pd.to_datetime(df[col], errors="coerce")
    return col

# ---- Função para carregar múltiplos CSVs ----
@st.cache_data(show_spinner=False)
def load_all_csvs_from_folder(folder_path: str, 
                            sep_opt: Optional[str] = None,
                            decimal_opt: str = ",",
                            skiprows: int = 0,
                            encodings: List[str] = ENCODINGS_BR) -> pd.DataFrame:
    """
    Carrega todos os arquivos CSV de uma pasta e concatena em um único DataFrame
    """
    # Encontrar todos os arquivos CSV na pasta
    csv_pattern = os.path.join(folder_path, "*.csv")
    csv_files = glob.glob(csv_pattern)
    
    if not csv_files:
        raise FileNotFoundError(f"Nenhum arquivo CSV encontrado em: {folder_path}")
    
    st.info(f"Encontrados {len(csv_files)} arquivos CSV na pasta")
    
    all_dfs = []
    file_info = []
    
    for i, csv_file in enumerate(csv_files):
        try:
            st.write(f"📁 Carregando: {os.path.basename(csv_file)}")
            df_temp, enc_used, sep_used = load_csv_simple(
                csv_file, sep_opt=sep_opt, decimal_opt=decimal_opt,
                skiprows=skiprows, encodings=encodings
            )
            
            # Adicionar coluna com nome do arquivo de origem
            df_temp['arquivo_origem'] = os.path.basename(csv_file)
            
            all_dfs.append(df_temp)
            file_info.append({
                'arquivo': os.path.basename(csv_file),
                'linhas': len(df_temp),
                'colunas': len(df_temp.columns),
                'encoding': enc_used,
                'separador': sep_used
            })
            
        except Exception as e:
            st.warning(f"⚠️ Erro ao carregar {csv_file}: {str(e)}")
            continue
    
    if not all_dfs:
        raise RuntimeError("Nenhum arquivo CSV pôde ser carregado com sucesso")
    
    # Concatenar todos os DataFrames
    final_df = pd.concat(all_dfs, ignore_index=True)
    
    # Mostrar informações dos arquivos carregados
    st.success(f"✅ Carregados {len(all_dfs)} arquivos com {len(final_df):,} registros no total")
    
    # Tabela de informações dos arquivos
    info_df = pd.DataFrame(file_info)
    with st.expander("📊 Informações dos arquivos carregados"):
        st.dataframe(info_df, use_container_width=True)
    
    return final_df

# ---- fallback de gradient (se não houver matplotlib) ----
def style_heatmap(df: pd.DataFrame, cmap: str = "Blues"):
    try:
        import matplotlib  # noqa
        return df.style.background_gradient(cmap=cmap)
    except Exception:
        return df

# --------------- Mapeamentos ---------------
UF_MAPPING = {
    "Rio de Janeiro": "RJ", "Mato Grosso": "MT", "Santa Catarina": "SC", "São Paulo": "SP",
    "Distrito Federal": "DF", "Zerado": "Zerado", "Pernambuco": "PE", "Mato Grosso do Sul": "MS",
    "Amazonas": "AM", "Paraná": "PR", "Parana": "PR", "PARANÁ": "PR", "PARANA": "PR",
    "Ceará": "CE", "Ceara": "CE", "Minas Gerais": "MG", 
    "Rio Grande do Sul": "RS", "Bahia": "BA", "Alagoas": "AL", "Pará": "PA", "Para": "PA",
    "Espírito Santo": "ES", "Espirito Santo": "ES", "Tocantins": "TO", "Paraíba": "PB", "Paraiba": "PB",
    "Sergipe": "SE", "Piauí": "PI", "Piaui": "PI", "Rio Grande do Norte": "RN", "Maranhão": "MA", 
    "Maranhao": "MA", "Goiás": "GO", "Goias": "GO", "Rondônia": "RO", "Rondonia": "RO", 
    "Amapá": "AP", "Amapa": "AP", "Roraima": "RR", "Acre": "AC",
    # Adicionar variações comuns
    "São Paulo": "SP", "Sao Paulo": "SP", "SANTA CATARINA": "SC", "RIO DE JANEIRO": "RJ",
    "RIO GRANDE DO SUL": "RS", "MINAS GERAIS": "MG", "BAHIA": "BA", "CEARÁ": "CE", "CEARA": "CE",
    "PARÁ": "PA", "PARA": "PA", "ESPÍRITO SANTO": "ES", "ESPIRITO SANTO": "ES", "GOIÁS": "GO",
    "GOIAS": "GO", "MARANHÃO": "MA", "MARANHAO": "MA", "PIAUÍ": "PI", "PIAUI": "PI",
    "PARAÍBA": "PB", "PARAIBA": "PB", "RONDÔNIA": "RO", "RONDONIA": "RO", "AMAPÁ": "AP",
    "AMAPA": "AP", "ACRE": "AC", "ALAGOAS": "AL", "SERGIPE": "SE", "TOCANTINS": "TO",
    "RIO GRANDE DO NORTE": "RN", "MATO GROSSO": "MT", "MATO GROSSO DO SUL": "MS",
    "DISTRITO FEDERAL": "DF", "PERNAMBUCO": "PE"
}

def create_municipio_mapping_file():
    """Cria o arquivo de mapeamento de municípios"""
    mapping_data = """000000-Ignorado	Ignorado
110002-Ariquemes	Ariquemes
110003-Cabixi	Cabixi
110004-Cacoal	Cacoal
110005-Cerejeiras	Cerejeiras
110006-Colorado do O	Colorado do Oeste
110007-Corumbiara	Corumbiara
110009-Espigão D'Oes	Espigão d'Oeste
110011-Jaru	Jaru
110012-Ji-Paraná	Ji Paraná
110013-Machadinho D'	Machadinho d'Oeste
110015-Ouro Preto do	Ouro Preto do Oeste
110018-Pimenta Bueno	Pimenta Bueno
110020-Porto Velho	Porto Velho
110025-Presidente Mé	Presidente Médici
110028-Rolim de Mour	Rolim de Moura
110030-Vilhena	Vilhena
110032-São Miguel do	São Miguel do Araguaia
110033-Nova Mamoré	Nova Mamoré
110034-Alvorada D'Oe	Alvorada d'Oeste
110045-Buritis-Ro	Buritis Ro
110080-Candeias do J	Candeias do Jamari
110092-Chupinguaia	Chupinguaia
110130-Mirante da Se	Mirante da Serra
110140-Monte Negro	Monte Negro
110143-Nova União-Ro	Nova União Ro
110145-Parecis	Parecis
110150-Seringueiras	Seringueiras
110170-Urupá	Urupá
110175-Vale do Anari	Vale do Anari
110180-Vale do Paraí	Vale do Paraíso
120010-Brasiléia	Brasiléia
120020-Cruzeiro do S	Cruzeiro do Sul
120025-Epitaciolândi	Epitaciolândi
120040-Rio Branco-Ac	Rio Branco Ac
120045-Senador Guiom	Senador Guiomard
120050-Sena Madureir	Sena Madureira
120060-Tarauacá	Tarauacá
120070-Xapuri	Xapuri
130014-Apuí	Apuí
130030-Autazes	Autazes
130040-Barcelos	Barcelos
130120-Coari	Coari
130170-Humaitá-Am	Humaitá Am
130185-Iranduba	Iranduba
130190-Itacoatiara	Itacoatiara
130250-Manacapuru	Manacapuru
130260-Manaus	Manaus
130300-Nhamundá	Nhamundá
130330-Novo Aripuanã	Novo Aripuanã
130340-Parintins	Parintins
130353-Presidente Fi	Presidente Figueiredo
130356-Rio Preto da	Rio Preto da
140010-Boa Vista-Rr	Boa Vista Rr
140017-Cantá	Cantá
140020-Caracaraí	Caracaraí
140050-São João da B	São João da B
150010-Abaetetuba	Abaetetuba
150020-Acará	Acará
150034-Água Azul do	Água Azul do Norte
150040-Alenquer	Alenquer
150060-Altamira	Altamira
150080-Ananindeua	Ananindeua
150100-Aveiro	Aveiro
150125-Bannach	Bannach
150130-Barcarena	Barcarena
150140-Belém-Pa	Belém Pa
150145-Belterra	Belterra
150150-Benevides	Benevides
150170-Bragança	Bragança
150172-Brasil Novo	Brasil Novo
150180-Breves	Breves
150215-Canaã dos Car	Canaã dos Carajás
150220-Capanema-Pa	Capanema Pa
150230-Capitão Poço	Capitão Poço
150240-Castanhal	Castanhal
150260-Colares	Colares
150270-Conceição do	Conceição do Araguaia
150275-Concórdia do	Concórdia do Pará
150276-Cumaru do Nor	Cumaru do Norte
150277-Curionópolis	Curionópolis
150290-Curuçá	Curuçá
150293-Dom Eliseu	Dom Eliseu
150345-Ipixuna do Pa	Ipixuna do Pa
150360-Itaituba	Itaituba
150375-Jacareacanga	Jacareacanga
150390-Juruti	Juruti
150420-Marabá	Marabá
150430-Maracanã	Maracanã
150442-Marituba	Marituba
150470-Moju	Moju
150480-Monte Alegre-	Monte Alegre
150495-Nova Esperanç	Nova Esperanç
150503-Novo Progress	Novo Progresso
150510-Óbidos	Óbidos
150530-Oriximiná	Oriximiná
150543-Ourilândia do	Ourilândia do
150550-Paragominas	Paragominas
150553-Parauapebas	Parauapebas
150563-Piçarra	Piçarra
150613-Redenção-Pa	Redenção Pa
150616-Rio Maria	Rio Maria
150618-Rondon do Par	Rondon do Pará
150619-Rurópolis	Rurópolis
150620-Salinópolis	Salinópolis
150630-Salvaterra	Salvaterra
150650-Santa Isabel	Santa Isabel
150658-Santa Maria d	Santa Maria da Vitória
150660-Santa Maria d	Santa Maria da Vitória
150670-Santana do Ar	Santana do Araguaia
150680-Santarém-Pa	Santarém Pa
150700-Santo Antônio	Santo Antônio de Jesus
150715-São Domingos	São Domingos
150720-São Domingos	São Domingos
150730-São Félix do	São Félix do Xingu
150745-São Geraldo d	São Geraldo do Araguaia
150750-São João do A	São João do A
150760-São Miguel do	São Miguel do Araguaia
150795-Tailândia	Tailândia
150800-Tomé-Açu	Tomé Açu
150803-Tracuateua	Tracuateua
150808-Tucumã	Tucumã
150810-Tucuruí	Tucuruí
150812-Ulianópolis	Ulianópolis
150815-Uruará	Uruará
150820-Vigia	Vigia
150830-Viseu	Viseu
150840-Xinguara	Xinguara
160020-Calçoene	Calçoene
160030-Macapá	Macapá
160040-Mazagão	Mazagão
160050-Oiapoque	Oiapoque
160060-Santana-Ap	Santana Ap
170030-Aguiarnópolis	Aguiarnópolis
170070-Alvorada-To	Alvorada To
170210-Araguaína	Araguaína
170220-Araguatins	Araguatins
170240-Arraias	Arraias
170255-Augustinópoli	Augustinópolis
170300-Babaçulândia	Babaçulândia
170320-Bernardo Sayã	Bernardo Sayã
170386-Cariri do Toc	Cariri do Toc
170550-Colinas do To	Colinas do Tocantins
170650-Darcinópolis	Darcinópolis
170700-Dianópolis	Dianópolis
170710-Divinópolis d	Divinópolis D
170755-Fátima-To	Fátima To
170820-Formoso do Ar	Formoso do Araguaia
170930-Guaraí	Guaraí
170950-Gurupi	Gurupi
171190-Lagoa da Conf	Lagoa da Confusão
171245-Luzinópolis	Luzinópolis
171320-Miracema do T	Miracema do Tocantins
171420-Natividade-To	Natividade To
171488-Nova Olinda-T	Nova Olinda T
171510-Novo Acordo	Novo Acordo
171610-Paraíso do To	Paraíso do Tocantins
171820-Porto Naciona	Porto Nacional
171850-Recursolândia	Recursolândia
171855-Riachinho-To	Riachinho To
172090-Taguatinga	Taguatinga
172100-Palmas-To	Palmas To
172120-Tocantinópoli	Tocantinópolis
172210-Xambioá	Xambioá
210005-Açailândia	Açailândia
210030-Aldeias Altas	Aldeias Altas
210100-Arari	Arari
210120-Bacabal	Bacabal
210125-Bacabeira	Bacabeira
210140-Balsas	Balsas
210150-Barão de Graj	Barão de Graj
210160-Barra do Cord	Barra do Cord
210170-Barreirinhas	Barreirinhas
210230-Buriti Bravo	Buriti Bravo
210280-Carolina	Carolina
210300-Caxias	Caxias
210320-Chapadinha	Chapadinha
210330-Codó	Codó
210340-Coelho Neto	Coelho Neto
210350-Colinas-Ma	Colinas Ma
210410-Fortaleza dos	Fortaleza dos Nogueiras
210455-Governador Ed	Governador Edison Lobão
210480-Grajaú	Grajaú
210515-Igarapé do Me	Igarapé do Me
210530-Imperatriz	Imperatriz
210540-Itapecuru Mir	Itapecuru Mirim
210542-Itinga do Mar	Itinga do Mar
210570-Lago da Pedra	Lago da Pedra
210610-Loreto	Loreto
210750-Paço do Lumia	Paço do Lumiar
210770-Paraibano	Paraibano
210845-Peritoró	Peritoró
210900-Porto Franco	Porto Franco
210910-Presidente Du	Presidente Du
210950-Riachão-Ma	Riachão Ma
210990-Santa Inês-Ma	Santa Inês Ma
211000-Santa Luzia-M	Santa Luzia M
211030-Santo Antônio	Santo Antônio de Jesus
211050-São Bento-Ma	São Bento Ma
211120-São José de R	São José do Rio Pardo
211130-São Luís	São Luís
211150-São Mateus do	São Mateus do Sul
211160-São Raimundo	São Raimundo
211200-Tasso Fragoso	Tasso Fragoso
211210-Timbiras	Timbiras
211220-Timon	Timon
211230-Tuntum	Tuntum
220020-Água Branca-P	Água Branca P
220040-Altos	Altos
220140-Barro Duro	Barro Duro
220190-Bom Jesus-Pi	Bom Jesus Pi
220208-Cajueiro da P	Cajueiro da Praia
220220-Campo Maior	Campo Maior
220230-Canto do Buri	Canto do Buriti
220290-Corrente	Corrente
220330-Demerval Lobã	Demerval Lobão
220390-Floriano	Floriano
220450-Guadalupe	Guadalupe
220530-Jerumenha	Jerumenha
220557-Lagoa de São	Lagoa de São
220570-Luís Correia	Luís Correia
220620-Miguel Alves	Miguel Alves
220700-Oeiras	Oeiras
220770-Parnaíba	Parnaíba
220800-Picos	Picos
220840-Piripiri	Piripiri
221060-São Raimundo	São Raimundo
221100-Teresina	Teresina
221110-União	União
221120-Uruçuí	Uruçuí
221130-Valença do Pi	Valença do Pi
230020-Acaraú	Acaraú
230030-Acopiara	Acopiara
230075-Amontada	Amontada
230100-Aquiraz	Aquiraz
230110-Aracati	Aracati
230190-Barbalha	Barbalha
230210-Baturité	Baturité
230220-Beberibe	Beberibe
230240-Boa Viagem	Boa Viagem
230250-Brejo Santo	Brejo Santo
230260-Camocim	Camocim
230280-Canindé	Canindé
230320-Caririaçu	Caririaçu
230350-Cascavel-Ce	Cascavel Ce
230370-Caucaia	Caucaia
230395-Chorozinho	Chorozinho
230410-Crateús	Crateús
230420-Crato	Crato
230428-Eusébio	Eusébio
230435-Forquilha	Forquilha
230440-Fortaleza	Fortaleza
230470-Granja	Granja
230495-Guaiúba	Guaiúba
230500-Guaraciaba do	Guaraciaba do
230523-Horizonte	Horizonte
230535-Icapuí	Icapuí
230540-Icó	Icó
230550-Iguatu-Ce	Iguatu Ce
230570-Ipaumirim	Ipaumirim
230610-Irauçuba	Irauçuba
230620-Itaiçaba	Itaiçaba
230625-Itaitinga	Itaitinga
230630-Itapagé	Itapagé
230640-Itapipoca	Itapipoca
230690-Jaguaribe	Jaguaribe
230725-Jijoca de Jer	Jijoca de Jer
230730-Juazeiro do N	Juazeiro do Norte
230740-Jucás	Jucás
230760-Limoeiro do N	Limoeiro do Norte
230765-Maracanaú	Maracanaú
230770-Maranguape	Maranguape
230780-Marco	Marco
230800-Massapê	Massapê
230810-Mauriti	Mauriti
230830-Milagres-Ce	Milagres Ce
230840-Missão Velha	Missão Velha
230850-Mombaça	Mombaça
230870-Morada Nova	Morada Nova
230960-Pacajus	Pacajus
230970-Pacatuba-Ce	Pacatuba Ce
230990-Pacujá	Pacujá
231000-Palhano	Palhano
231020-Paracuru	Paracuru
231025-Paraipaba	Paraipaba
231080-Pereiro	Pereiro
231130-Quixadá	Quixadá
231140-Quixeramobim	Quixeramobim
231150-Quixeré	Quixeré
231160-Redenção-Ce	Redenção Ce
231180-Russas	Russas
231220-Santa Quitéri	Santa Quitéri
231230-São Benedito	São Benedito
231240-São Gonçalo d	São Gonçalo do Sapucaí
231270-Senador Pompe	Senador Pompeu
231290-Sobral	Sobral
231330-Tauá	Tauá
231340-Tianguá	Tianguá
231350-Trairi	Trairi
231380-Uruburetama	Uruburetama
231390-Uruoca	Uruoca
231400-Várzea Alegre	Várzea Alegre
231410-Viçosa do Cea	Viçosa do Cea
240020-Açu	Açu
240030-Afonso Bezerr	Afonso Bezerra
240080-Angicos	Angicos
240100-Apodi	Apodi
240110-Areia Branca-	Areia Branca
240120-Arês	Arês
240140-Baía Formosa	Baía Formosa
240145-Baraúna-Rn	Baraúna Rn
240170-Bom Jesus-Rn	Bom Jesus Rn
240230-Caraúbas-Rn	Caraúbas Rn
240260-Ceará-Mirim	Ceará Mirim
240270-Cerro Corá	Cerro Corá
240310-Currais Novos	Currais Novos
240325-Parnamirim-Rn	Parnamirim Rn
240380-Florânia	Florânia
240410-Galinhos	Galinhos
240420-Goianinha	Goianinha
240430-Governador Di	Governador Di
240440-Grossos	Grossos
240450-Guamaré	Guamaré
240530-Januário Cicc	Januário Cicc
240610-Jucurutu	Jucurutu
240710-Macaíba	Macaíba
240725-Major Sales	Major Sales
240780-Monte Alegre-	Monte Alegre
240800-Mossoró	Mossoró
240810-Natal	Natal
240830-Nova Cruz	Nova Cruz
240890-Parelhas	Parelhas
240940-Pau dos Ferro	Pau dos Ferro
241030-Presidente Ju	Presidente Prudente
241160-São Bento do	São Bento do Sul
241200-São Gonçalo d	São Gonçalo do Sapucaí
241220-São José de M	São José de Mipibu
241240-São José do S	São José do Sabugi
241250-São Miguel	São Miguel
241260-São Paulo do	São Paulo do Potengi
241310-Senador Elói	Senador Elói
241340-Serra Negra d	Serra Negra do Norte
241400-Tangará-Rn	Tangará Rn
241420-Tibau do Sul	Tibau do Sul
241440-Touros	Touros
241460-Upanema	Upanema
250110-Areia	Areia
250120-Areial	Areial
250180-Bayeux	Bayeux
250215-Boa Vista-Pb	Boa Vista Pb
250300-Caaporã	Caaporã
250320-Cabedelo	Cabedelo
250370-Cajazeiras	Cajazeiras
250400-Campina Grand	Campina Grande
250430-Catolé do Roc	Catolé do Rocha
250460-Conde-Pb	Conde Pb
250580-Duas Estradas	Duas Estradas
250600-Esperança	Esperança
250630-Guarabira	Guarabira
250680-Ingá	Ingá
250700-Itaporanga-Pb	Itaporanga Pb
250750-João Pessoa	João Pessoa
250830-Lagoa Seca	Lagoa Seca
250860-Lucena	Lucena
250890-Mamanguape	Mamanguape
250920-Massaranduba-	Massaranduba
250970-Monteiro	Monteiro
251080-Patos	Patos
251090-Paulista-Pb	Paulista Pb
251120-Pedras de Fog	Pedras de Fogo
251150-Pilar-Pb	Pilar Pb
251200-Pocinhos	Pocinhos
251230-Princesa Isab	Princesa Isabel
251240-Puxinanã	Puxinanã
251250-Queimadas-Pb	Queimadas Pb
251290-Rio Tinto	Rio Tinto
251370-Santa Rita-Pb	Santa Rita Pb
251390-São Bento-Pb	São Bento Pb
251400-São João do C	São João do C
251450-São José de P	São José de P
251500-São Miguel de	São Miguel do Guamá
251610-Soledade-Pb	Soledade Pb
251620-Sousa	Sousa
251630-Sumé	Sumé
260005-Abreu e Lima	Abreu e Lima
260010-Afogados da I	Afogados da Ingazeira
260040-Água Preta	Água Preta
260070-Aliança	Aliança
260110-Araripina	Araripina
260120-Arcoverde	Arcoverde
260140-Barreiros	Barreiros
260170-Belo Jardim	Belo Jardim
260190-Bezerros	Bezerros
260230-Bonito-Pe	Bonito Pe
260250-Brejinho-Pe	Brejinho Pe
260280-Buíque	Buíque
260290-Cabo de Santo	Cabo de Santo Agostinho
260345-Camaragibe	Camaragibe
260360-Camutanga	Camutanga
260370-Canhotinho	Canhotinho
260400-Carpina	Carpina
260410-Caruaru	Caruaru
260500-Cupira	Cupira
260510-Custódia	Custódia
260520-Escada	Escada
260540-Feira Nova-Pe	Feira Nova Pe
260570-Floresta-Pe	Floresta Pe
260600-Garanhuns	Garanhuns
260610-Glória do Goi	Glória do Goitá
260620-Goiana	Goiana
260640-Gravatá	Gravatá
260680-Igarassu	Igarassu
260690-Iguaraci	Iguaraci
260720-Ipojuca	Ipojuca
260730-Ipubi	Ipubi
260775-Itapissuma	Itapissuma
260790-Jaboatão dos	Jaboatão dos Guararapes
260800-Jataúba	Jataúba
260840-Jurema-Pe	Jurema Pe
260850-Lagoa do Itae	Lagoa do Itaenga
260875-Lagoa Grande-	Lagoa Grande
260890-Limoeiro	Limoeiro
260940-Moreno	Moreno
260950-Nazaré da Mat	Nazaré da Mata
260960-Olinda	Olinda
260990-Ouricuri	Ouricuri
261000-Palmares	Palmares
261050-Passira	Passira
261060-Paudalho	Paudalho
261070-Paulista-Pe	Paulista Pe
261090-Pesqueira	Pesqueira
261110-Petrolina	Petrolina
261130-Pombos	Pombos
261140-Primavera-Pe	Primavera Pe
261160-Recife	Recife
261170-Riacho das Al	Riacho das Almas
261180-Ribeirão	Ribeirão
261190-Rio Formoso	Rio Formoso
261220-Salgueiro	Salgueiro
261250-Santa Cruz do	Santa Cruz do Sul
261260-Santa Maria d	Santa Maria da Vitória
261300-São Bento do	São Bento do Sul
261310-São Caitano	São Caetano
261340-São José da C	São José da C
261350-São José do B	São José do Barreiro
261360-São José do E	São José do Egito
261370-São Lourenço	São Lourenço
261380-São Vicente F	São Vicente F
261390-Serra Talhada	Serra Talhada
261420-Sirinhaém	Sirinhaém
261450-Surubim	Surubim
261485-Tamandaré	Tamandaré
261530-Timbaúba	Timbaúba
261630-Vicência	Vicência
261640-Vitória de Sa	Vitória de Santo Antão
270030-Arapiraca	Arapiraca
270040-Atalaia-Al	Atalaia Al
270050-Barra de Sant	Barra de Sant
270060-Barra de São	Barra de São Francisco
270130-Cajueiro	Cajueiro
270140-Campo Alegre-	Campo Alegre
270230-Coruripe	Coruripe
270320-Igreja Nova	Igreja Nova
270360-Japaratinga	Japaratinga
270375-Jequiá da Pra	Jequiá da Pra
270410-Lagoa da Cano	Lagoa da Cano
270430-Maceió	Maceió
270450-Maragogi	Maragogi
270470-Marechal Deod	Marechal Deodoro
270510-Matriz de Cam	Matriz de Cam
270630-Palmeira dos	Palmeira dos Índios
270670-Penedo	Penedo
270690-Pilar-Al	Pilar Al
270710-Piranhas-Al	Piranhas Al
270750-Porto Real do	Porto Real do Colégio
270770-Rio Largo	Rio Largo
270790-Santa Luzia d	Santa Luzia D
270800-Santana do Ip	Santana do Ipanema
270830-São José da L	São José da Lapa
270840-São José da T	São José da T
270850-São Luís do Q	São Luís do Quitunde
270860-São Miguel do	São Miguel do Araguaia
270880-São Sebastião	São Sebastião
270930-União dos Pal	União dos Palmares
280030-Aracaju	Aracaju
280060-Barra dos Coq	Barra dos Coqueiros
280100-Campo do Brit	Campo do Brit
280130-Capela-Se	Capela Se
280150-Carmópolis	Carmópolis
280210-Estância	Estância
280230-Frei Paulo	Frei Paulo
280290-Itabaiana-Se	Itabaiana Se
280300-Itabaianinha	Itabaianinha
280330-Japaratuba	Japaratuba
280350-Lagarto	Lagarto
280360-Laranjeiras	Laranjeiras
280400-Maruim	Maruim
280440-Neópolis	Neópolis
280450-Nossa Senhora	Nossa Senhora
280460-Nossa Senhora	Nossa Senhora
280480-Nossa Senhora	Nossa Senhora
280550-Poço Verde	Poço Verde
280570-Propriá	Propriá
280600-Ribeirópolis	Ribeirópolis
280610-Rosário do Ca	Rosário do Catete
280670-São Cristóvão	São Cristóvão
280710-Simão Dias	Simão Dias
290030-Acajutiba	Acajutiba
290050-Érico Cardoso	Érico Cardoso
290070-Alagoinhas	Alagoinhas
290100-Amargosa	Amargosa
290110-Amélia Rodrig	Amélia Rodrigues
290135-Andorinha	Andorinha
290200-Aracatu	Aracatu
290220-Aramari	Aramari
290280-Barra da Esti	Barra da Estiva
290290-Barra do Choç	Barra do Choça
290320-Barreiras	Barreiras
290327-Barrocas	Barrocas
290340-Belmonte-Ba	Belmonte Ba
290390-Bom Jesus da	Bom Jesus da Lapa
290440-Brejolândia	Brejolândia
290460-Brumado	Brumado
290470-Buerarema	Buerarema
290490-Cachoeira	Cachoeira
290500-Caculé	Caculé
290520-Caetité	Caetité
290570-Camaçari	Camaçari
290580-Camamu	Camamu
290600-Campo Formoso	Campo Formoso
290650-Candeias-Ba	Candeias Ba
290685-Capela do Alt	Capela do Alto
290687-Capim Grosso	Capim Grosso
290720-Casa Nova	Casa Nova
290730-Castro Alves	Castro Alves
290750-Catu	Catu
290780-Cícero Dantas	Cícero Dantas
290810-Cocos	Cocos
290820-Conceição da	Conceição da Barra
290830-Conceição do	Conceição do Araguaia
290840-Conceição do	Conceição do Araguaia
290850-Conceição do	Conceição do Araguaia
290860-Conde-Ba	Conde Ba
290890-Coração de Ma	Coração de Ma
290930-Correntina	Correntina
290940-Cotegipe	Cotegipe
290980-Cruz das Alma	Cruz das Almas
291005-Dias D'Ávila	Dias d'Ávila
291020-Dom Macedo Co	Dom Macedo Costa
291060-Esplanada	Esplanada
291070-Euclides da C	Euclides da Cunha
291072-Eunápolis	Eunápolis
291080-Feira de Sant	Feira de Santana
291085-Filadélfia-Ba	Filadélfia Ba
291120-Gandu	Gandu
291130-Gentio do Our	Gentio do Ouro
291160-Governador Ma	Governador Mangabeira
291170-Guanambi	Guanambi
291190-Iaçu	Iaçu
291220-Ibicoara	Ibicoara
291270-Ibirapitanga	Ibirapitanga
291280-Ibirapuã	Ibirapuã
291290-Ibirataia	Ibirataia
291320-Ibotirama	Ibotirama
291345-Igrapiúna	Igrapiúna
291350-Iguaí	Iguaí
291360-Ilhéus	Ilhéus
291390-Ipiaú	Ipiaú
291400-Ipirá	Ipirá
291460-Irecê	Irecê
291470-Itaberaba	Itaberaba
291480-Itabuna	Itabuna
291490-Itacaré	Itacaré
291560-Itamaraju	Itamaraju
291610-Itaparica	Itaparica
291640-Itapetinga	Itapetinga
291650-Itapicuru	Itapicuru
291680-Itarantim	Itarantim
291685-Itatim	Itatim
291730-Ituberá	Ituberá
291735-Jaborandi-Ba	Jaborandi Ba
291750-Jacobina	Jacobina
291760-Jaguaquara	Jaguaquara
291770-Jaguarari	Jaguarari
291800-Jequié	Jequié
291840-Juazeiro	Juazeiro
291920-Lauro de Frei	Lauro de Freitas
291950-Livramento de	Livramento de Nossa Senhora
291955-Luís Eduardo	Luís Eduardo Magalhães
291980-Macaúbas	Macaúbas
292000-Maiquinique	Maiquinique
292040-Manoel Vitori	Manoel Vitori
292070-Maraú	Maraú
292100-Mata de São J	Mata de São João
292110-Medeiros Neto	Medeiros Neto
292140-Mirangaba	Mirangaba
292150-Monte Santo	Monte Santo
292190-Mucugê	Mucugê
292200-Mucuri	Mucuri
292230-Muritiba	Muritiba
292240-Mutuípe	Mutuípe
292250-Nazaré-Ba	Nazaré Ba
292300-Nova Viçosa	Nova Viçosa
292335-Ourolândia	Ourolândia
292350-Palmeiras	Palmeiras
292360-Paramirim	Paramirim
292370-Paratinga	Paratinga
292400-Paulo Afonso	Paulo Afonso
292430-Piatã	Piatã
292467-Piraí do Nort	Piraí do Norte
292490-Planaltino	Planaltino
292500-Planalto-Ba	Planalto Ba
292510-Poções	Poções
292520-Pojuca	Pojuca
292530-Porto Seguro	Porto Seguro
292550-Prado	Prado
292560-Presidente Du	Presidente Du
292580-Queimadas-Ba	Queimadas Ba
292600-Remanso	Remanso
292630-Riachão do Ja	Riachão do Jacuípe
292650-Ribeira do Am	Ribeira do Am
292660-Ribeira do Po	Ribeira do Pombal
292700-Rio Real	Rio Real
292720-Ruy Barbosa-B	Ruy Barbosa B
292740-Salvador	Salvador
292770-Santa Cruz Ca	Santa Cruz Cabrália
292810-Santa Maria d	Santa Maria da Vitória
292840-Santa Rita de	Santa Rita de Cássia
292850-Santa Teresin	Santa Teresin
292860-Santo Amaro	Santo Amaro
292870-Santo Antônio	Santo Antônio de Jesus
292880-Santo Estêvão	Santo Estêvão
292890-São Desidério	São Desidério
292920-São Francisco	São Francisco do Sul
292930-São Gonçalo d	São Gonçalo do Sapucaí
292940-São Miguel da	São Miguel das Missões
292950-São Sebastião	São Sebastião
292960-Sapeaçu	Sapeaçu
292970-Sátiro Dias	Sátiro Dias
292990-Seabra	Seabra
293010-Senhor do Bon	Senhor do Bonfim
293050-Serrinha-Ba	Serrinha Ba
293070-Simões Filho	Simões Filho
293075-Sítio do Mato	Sítio do Mato
293100-Tanhaçu	Tanhaçu
293135-Teixeira de F	Teixeira de Freitas
293140-Teodoro Sampa	Teodoro Sampaio
293150-Teofilândia	Teofilândia
293170-Terra Nova-Ba	Terra Nova Ba
293180-Tremedal	Tremedal
293190-Tucano	Tucano
293210-Ubaíra	Ubaíra
293250-Una	Una
293280-Utinga	Utinga
293290-Valença-Ba	Valença Ba
293300-Valente	Valente
293310-Várzea do Poç	Várzea do Poç
293320-Vera Cruz-Ba	Vera Cruz Ba
293330-Vitória da Co	Vitória da Conquista
293360-Xique-Xique	Xique Xique
310020-Abaeté	Abaeté
310030-Abre Campo	Abre Campo
310070-Água Comprida	Água Comprida
310090-Águas Formosa	Águas Formosa
310100-Águas Vermelh	Águas Vermelh
310110-Aimorés	Aimorés
310130-Alagoa	Alagoa
310150-Além Paraíba	Além Paraíba
310160-Alfenas	Alfenas
310170-Almenara	Almenara
310180-Alpercata	Alpercata
310190-Alpinópolis	Alpinópolis
310230-Alvinópolis	Alvinópolis
310260-Andradas	Andradas
310280-Andrelândia	Andrelândia
310290-Antônio Carlo	Antônio Carlos
310300-Antônio Dias	Antônio Dias
310340-Araçuaí	Araçuaí
310350-Araguari	Araguari
310375-Araporã	Araporã
310390-Araújos	Araújos
310400-Araxá	Araxá
310410-Arceburgo	Arceburgo
310420-Arcos	Arcos
310450-Arinos	Arinos
310460-Astolfo Dutra	Astolfo Dutra
310480-Augusto de Li	Augusto de Li
310490-Baependi	Baependi
310500-Baldim	Baldim
310510-Bambuí	Bambuí
310530-Bandeira do S	Bandeira do Sul
310540-Barão de Coca	Barão de Cocais
310560-Barbacena	Barbacena
310590-Barroso	Barroso
310600-Bela Vista de	Bela Vista de Goiás
310620-Belo Horizont	Belo Horizonte
310630-Belo Oriente	Belo Oriente
310640-Belo Vale	Belo Vale
310670-Betim	Betim
310690-Bicas	Bicas
310710-Boa Esperança	Boa Esperança do Iguaçu
310730-Bocaiúva	Bocaiúva
310740-Bom Despacho	Bom Despacho
310750-Bom Jardim de	Bom Jardim de
310770-Bom Jesus do	Bom Jesus do
310780-Bom Jesus do	Bom Jesus do
310800-Bom Sucesso-M	Bom Sucesso M
310820-Bonfinópolis	Bonfinópolis
310830-Borda da Mata	Borda da Mata
310840-Botelhos	Botelhos
310850-Botumirim	Botumirim
310855-Brasilândia d	Brasilândia do Sul
310860-Brasília de M	Brasília de M
310900-Brumadinho	Brumadinho
310910-Bueno Brandão	Bueno Brandão
310930-Buritis-Mg	Buritis Mg
310940-Buritizeiro	Buritizeiro
310950-Cabo Verde	Cabo Verde
310970-Cachoeira de	Cachoeira de Minas
310990-Caetanópolis	Caetanópolis
311000-Caeté	Caeté
311030-Caldas	Caldas
311050-Camanducaia	Camanducaia
311060-Cambuí	Cambuí
311100-Campestre-Mg	Campestre Mg
311110-Campina Verde	Campina Verde
311120-Campo Belo	Campo Belo
311140-Campo Florido	Campo Florido
311160-Campos Gerais	Campos Gerais
311180-Canápolis-Mg	Canápolis Mg
311200-Candeias-Mg	Candeias Mg
311205-Cantagalo-Mg	Cantagalo Mg
311220-Capela Nova	Capela Nova
311230-Capelinha	Capelinha
311250-Capim Branco	Capim Branco
311260-Capinópolis	Capinópolis
311290-Caputira	Caputira
311300-Caraí	Caraí
311320-Carandaí	Carandaí
311330-Carangola	Carangola
311340-Caratinga	Caratinga
311350-Carbonita	Carbonita
311360-Careaçu	Careaçu
311370-Carlos Chagas	Carlos Chagas
311390-Carmo da Cach	Carmo da Cachoeira
311400-Carmo da Mata	Carmo da Mata
311410-Carmo de Mina	Carmo de Mina
311420-Carmo do Caju	Carmo do Cajuru
311430-Carmo do Para	Carmo do Paranaíba
311440-Carmo do Rio	Carmo do Rio Claro
311450-Carmópolis de	Carmópolis de Minas
311455-Carneirinho	Carneirinho
311510-Cássia	Cássia
311530-Cataguases	Cataguases
311550-Caxambu	Caxambu
311580-Centralina	Centralina
311590-Chácara	Chácara
311640-Claraval	Claraval
311660-Cláudio	Cláudio
311670-Coimbra	Coimbra
311690-Comendador Go	Comendador Gomes
311730-Conceição das	Conceição das Alagoas
311750-Conceição do	Conceição do Araguaia
311760-Conceição do	Conceição do Araguaia
311770-Conceição do	Conceição do Araguaia
311780-Conceição dos	Conceição dos Ouros
311787-Confins	Confins
311790-Congonhal	Congonhal
311800-Congonhas	Congonhas
311830-Conselheiro L	Conselheiro Lafaiete
311840-Conselheiro P	Conselheiro Pena
311850-Consolação	Consolação
311860-Contagem	Contagem
311870-Coqueiral	Coqueiral
311890-Cordisburgo	Cordisburgo
311910-Corinto	Corinto
311930-Coromandel	Coromandel
311940-Coronel Fabri	Coronel Fabriciano
311960-Coronel Pache	Coronel Pacheco
311970-Coronel Xavie	Coronel Xavier Chaves
311980-Córrego Danta	Córrego Danta
311990-Córrego do Bo	Córrego do Bom Jesus
311995-Córrego Fundo	Córrego Fundo
312080-Cruzília	Cruzília
312090-Curvelo	Curvelo
312120-Delfinópolis	Delfinópolis
312125-Delta	Delta
312140-Desterro de E	Desterro de Entre Rios
312160-Diamantina	Diamantina
312230-Divinópolis	Divinópolis
312240-Divisa Nova	Divisa Nova
312290-Dona Euzébia	Dona Euzébia
312300-Dores de Camp	Dores de Campos
312320-Dores do Inda	Dores do Inda
312360-Elói Mendes	Elói Mendes
312380-Engenheiro Na	Engenheiro Navarro
312385-Entre Folhas	Entre Folhas
312390-Entre Rios de	Entre Rios de
312400-Ervália	Ervália
312410-Esmeraldas	Esmeraldas
312420-Espera Feliz	Espera Feliz
312430-Espinosa	Espinosa
312450-Estiva	Estiva
312490-Eugenópolis	Eugenópolis
312510-Extrema	Extrema
312550-São Gonçalo d	São Gonçalo do Sapucaí
312570-Felixlândia	Felixlândia
312590-Ferros	Ferros
312600-Florestal	Florestal
312610-Formiga	Formiga
312630-Fortaleza de	Fortaleza de
312660-Francisco Dum	Francisco Dumont
312670-Francisco Sá	Francisco Sá
312675-Franciscópoli	Franciscópoli
312700-Fronteira	Fronteira
312710-Frutal	Frutal
312738-Goianá	Goianá
312740-Gonçalves	Gonçalves
312760-Gouvêa	Gouvêa
312770-Governador Va	Governador Valadares
312780-Grão Mogol	Grão Mogol
312800-Guanhães	Guanhães
312810-Guapé	Guapé
312820-Guaraciaba-Mg	Guaraciaba Mg
312830-Guaranésia	Guaranésia
312840-Guarani	Guarani
312860-Guarda-Mor	Guarda Mor
312870-Guaxupé	Guaxupé
312880-Guidoval	Guidoval
312890-Guimarânia	Guimarânia
312900-Guiricema	Guiricema
312930-Iapu	Iapu
312940-Ibertioga	Ibertioga
312950-Ibiá	Ibiá
312970-Ibiraci	Ibiraci
312980-Ibirité	Ibirité
313010-Igarapé	Igarapé
313020-Igaratinga	Igaratinga
313030-Iguatama	Iguatama
313040-Ijaci	Ijaci
313050-Ilicínea	Ilicínea
313055-Imbé de Minas	Imbé de Minas
313060-Inconfidentes	Inconfidentes
313065-Indaiabira	Indaiabira
313070-Indianópolis-	Indianópolis
313080-Ingaí	Ingaí
313115-Ipaba	Ipaba
313130-Ipatinga	Ipatinga
313160-Iraí de Minas	Iraí de Minas
313170-Itabira	Itabira
313190-Itabirito	Itabirito
313210-Itacarambi	Itacarambi
313220-Itaguara	Itaguara
313240-Itajubá	Itajubá
313250-Itamarandiba	Itamarandiba
313270-Itambacuri	Itambacuri
313300-Itamonte	Itamonte
313310-Itanhandu	Itanhandu
313320-Itanhomi	Itanhomi
313330-Itaobim	Itaobim
313340-Itapagipe	Itapagipe
313350-Itapecerica	Itapecerica
313360-Itapeva-Mg	Itapeva Mg
313370-Itatiaiuçu	Itatiaiuçu
313375-Itaú de Minas	Itaú de Minas
313380-Itaúna	Itaúna
313400-Itinga	Itinga
313420-Ituiutaba	Ituiutaba
313440-Iturama	Iturama
313450-Itutinga	Itutinga
313460-Jaboticatubas	Jaboticatubas
313470-Jacinto	Jacinto
313490-Jacutinga-Mg	Jacutinga Mg
313500-Jaguaraçu	Jaguaraçu
313505-Jaíba	Jaíba
313507-Jampruca	Jampruca
313510-Janaúba	Janaúba
313520-Januária	Januária
313530-Japaraíba	Japaraíba
313540-Jeceaba	Jeceaba
313550-Jequeri	Jequeri
313570-Jequitibá	Jequitibá
313620-João Monlevad	João Monlevade
313630-João Pinheiro	João Pinheiro
313640-Joaquim Felíc	Joaquim Felício
313660-Nova União-Mg	Nova União Mg
313665-Juatuba	Juatuba
313670-Juiz de Fora	Juiz de Fora
313710-Lagamar	Lagamar
313720-Lagoa da Prat	Lagoa da Prata
313750-Lagoa Formosa	Lagoa Formosa
313753-Lagoa Grande-	Lagoa Grande
313760-Lagoa Santa-M	Lagoa Santa M
313770-Lajinha	Lajinha
313800-Laranjal-Mg	Laranjal Mg
313810-Lassance	Lassance
313820-Lavras	Lavras
313840-Leopoldina	Leopoldina
313860-Lima Duarte	Lima Duarte
313862-Limeira do Oe	Limeira do Oeste
313870-Luminárias	Luminárias
313880-Luz	Luz
313900-Machado	Machado
313910-Madre de Deus	Madre de Deus de Minas
313920-Malacacheta	Malacacheta
313940-Manhuaçu	Manhuaçu
313950-Manhumirim	Manhumirim
313960-Mantena	Mantena
314000-Mariana	Mariana
314015-Mário Campos	Mário Campos
314020-Maripá de Min	Maripá de Min
314030-Marliéria	Marliéria
314050-Martinho Camp	Martinho Camp
314053-Martins Soare	Martins Soares
314070-Mateus Leme	Mateus Leme
314080-Matias Barbos	Matias Barbosa
314090-Matipó	Matipó
314110-Matozinhos	Matozinhos
314120-Matutina	Matutina
314140-Medina	Medina
314180-Minas Novas	Minas Novas
314190-Minduri	Minduri
314220-Miraí	Miraí
314230-Moeda	Moeda
314260-Monsenhor Pau	Monsenhor Paulo
314280-Monte Alegre	Monte Alegre
314290-Monte Azul	Monte Azul
314300-Monte Belo	Monte Belo
314310-Monte Carmelo	Monte Carmelo
314320-Monte Santo d	Monte Santo de Minas
314330-Montes Claros	Montes Claros
314340-Monte Sião	Monte Sião
314350-Morada Nova d	Morada Nova de Minas
314390-Muriaé	Muriaé
314400-Mutum	Mutum
314410-Muzambinho	Muzambinho
314430-Nanuque	Nanuque
314440-Natércia	Natércia
314450-Nazareno	Nazareno
314460-Nepomuceno	Nepomuceno
314470-Nova Era	Nova Era
314480-Nova Lima	Nova Lima
314500-Nova Ponte	Nova Ponte
314510-Nova Resende	Nova Resende
314520-Nova Serrana	Nova Serrana
314530-Novo Cruzeiro	Novo Cruzeiro
314540-Olaria	Olaria
314545-Olhos-D'Água	Olhos D'água
314560-Oliveira	Oliveira
314590-Ouro Branco-M	Ouro Branco M
314600-Ouro Fino	Ouro Fino
314610-Ouro Preto	Ouro Preto
314625-Padre Carvalh	Padre Carvalho
314650-Pains	Pains
314690-Papagaios	Papagaios
314700-Paracatu	Paracatu
314710-Pará de Minas	Pará de Minas
314720-Paraguaçu	Paraguaçu
314730-Paraisópolis	Paraisópolis
314740-Paraopeba	Paraopeba
314750-Passabém	Passabém
314760-Passa Quatro	Passa Quatro
314770-Passa Tempo	Passa Tempo
314790-Passos	Passos
314800-Patos de Mina	Patos de Minas
314810-Patrocínio	Patrocínio
314820-Patrocínio do	Patrocínio do Muriaé
314870-Pedra Azul	Pedra Azul
314890-Pedra do Inda	Pedra do Indaiá
314910-Pedralva	Pedralva
314930-Pedro Leopold	Pedro Leopoldo
314970-Perdigão	Perdigão
314980-Perdizes	Perdizes
314990-Perdões	Perdões
315010-Piau	Piau
315015-Piedade de Ca	Piedade de Caratinga
315030-Piedade do Ri	Piedade do Rio Grande
315040-Piedade dos G	Piedade dos G
315070-Pirajuba	Pirajuba
315080-Piranga	Piranga
315090-Piranguçu	Piranguçu
315100-Piranguinho	Piranguinho
315110-Pirapetinga	Pirapetinga
315120-Pirapora	Pirapora
315140-Pitangui	Pitangui
315150-Piuí	Piuí
315160-Planura	Planura
315170-Poço Fundo	Poço Fundo
315180-Poços de Cald	Poços de Caldas
315200-Pompéu	Pompéu
315210-Ponte Nova	Ponte Nova
315220-Porteirinha	Porteirinha
315240-Poté	Poté
315250-Pouso Alegre	Pouso Alegre
315260-Pouso Alto	Pouso Alto
315270-Prados	Prados
315280-Prata-Mg	Prata Mg
315290-Pratápolis	Pratápolis
315340-Presidente Ol	Presidente Olegário
315360-Prudente de M	Presidente Prudente
315400-Raul Soares	Raul Soares
315410-Recreio	Recreio
315415-Reduto	Reduto
315420-Resende Costa	Resende Costa
315430-Resplendor	Resplendor
315440-Ressaquinha	Ressaquinha
315460-Ribeirão das	Ribeirão das Neves
315470-Ribeirão Verm	Ribeirão Vermelho
315480-Rio Acima	Rio Acima
315490-Rio Casca	Rio Casca
315500-Rio Doce	Rio Doce
315540-Rio Novo	Rio Novo
315550-Rio Paranaíba	Rio Paranaíba
315570-Rio Piracicab	Rio Piracicab
315580-Rio Pomba	Rio Pomba
315610-Ritápolis	Ritápolis
315620-Rochedo de Mi	Rochedo de Mi
315630-Rodeiro	Rodeiro
315650-Rubelita	Rubelita
315670-Sabará	Sabará
315690-Sacramento	Sacramento
315700-Salinas	Salinas
315720-Santa Bárbara	Santa Bárbara d'Oeste
315733-Santa Cruz de	Santa Cruz de
315760-Santa Fé de M	Santa Fé de Minas
315770-Santa Juliana	Santa Juliana
315780-Santa Luzia-M	Santa Luzia M
315800-Santa Maria d	Santa Maria da Vitória
315820-Santa Maria d	Santa Maria da Vitória
315830-Santana da Va	Santana da Vargem
315840-Santana de Ca	Santana de Ca
315850-Santana de Pi	Santana de Parnaíba
315895-Santana do Pa	Santana do Paraíso
315900-Santana do Ri	Santana do Riacho
315920-Santa Rita de	Santa Rita de Cássia
315935-Santa Rita de	Santa Rita de Cássia
315940-Santa Rita do	Santa Rita do Sapucaí
315960-Santa Rita do	Santa Rita do Sapucaí
315980-Santa Vitória	Santa Vitória
315990-Santo Antônio	Santo Antônio de Jesus
316000-Santo Antônio	Santo Antônio de Jesus
316010-Santo Antônio	Santo Antônio de Jesus
316040-Santo Antônio	Santo Antônio de Jesus
316070-Santos Dumont	Santos Dumont
316090-São Brás do S	São Brás do S
316100-São Domingos	São Domingos
316110-São Francisco	São Francisco do Sul
316120-São Francisco	São Francisco do Sul
316130-São Francisco	São Francisco do Sul
316150-São Geraldo	São Geraldo
316165-São Geraldo d	São Geraldo do Araguaia
316170-São Gonçalo d	São Gonçalo do Sapucaí
316180-São Gonçalo d	São Gonçalo do Sapucaí
316190-São Gonçalo d	São Gonçalo do Sapucaí
316200-São Gonçalo d	São Gonçalo do Sapucaí
316210-São Gotardo	São Gotardo
316220-São João Bati	São João Batista
316240-São João da P	São João da Ponta
316250-São João Del	São João del-Rei
316265-São João do P	São João do P
316270-São João do P	São João do P
316290-São João Nepo	São João Nepomuceno
316292-São Joaquim d	São Joaquim da Barra
316295-São José da L	São José da Lapa
316310-São José da V	São José da Varginha
316370-São Lourenço	São Lourenço
316380-São Miguel do	São Miguel do Araguaia
316440-São Sebastião	São Sebastião
316443-São Sebastião	São Sebastião
316460-São Sebastião	São Sebastião
316470-São Sebastião	São Sebastião
316490-São Sebastião	São Sebastião
316500-São Tiago	São Tiago
316510-São Tomás de	São Tomás de Aquino
316530-São Vicente d	São Vicente de Minas
316540-Sapucaí-Mirim	Sapucaí Mirim
316550-Sardoá	Sardoá
316553-Sarzedo	Sarzedo
316580-Senador José	Senador José Bento
316590-Senador Modes	Senador Modes
316610-Senhora do Po	Senhora do Porto
316670-Serra dos Aim	Serra dos Aimorés
316680-Serra do Sali	Serra do Salitre
316690-Serrania	Serrania
316700-Serranos	Serranos
316710-Serro	Serro
316720-Sete Lagoas	Sete Lagoas
316750-Simão Pereira	Simão Pereira
316780-Soledade de M	Soledade de Minas
316800-Taiobeiras	Taiobeiras
316810-Tapira-Mg	Tapira Mg
316850-Teixeiras	Teixeiras
316860-Teófilo Otoni	Teófilo Otoni
316870-Timóteo	Timóteo
316880-Tiradentes	Tiradentes
316900-Tocantins	Tocantins
316910-Toledo-Mg	Toledo Mg
316930-Três Corações	Três Corações
316935-Três Marias	Três Marias
316940-Três Pontas	Três Pontas
316960-Tupaciguara	Tupaciguara
316970-Turmalina-Mg	Turmalina Mg
316980-Turvolândia	Turvolândia
316990-Ubá	Ubá
317010-Uberaba	Uberaba
317020-Uberlândia	Uberlândia
317040-Unaí	Unaí
317047-Uruana de Min	Uruana de Min
317050-Urucânia	Urucânia
317070-Varginha	Varginha
317080-Várzea da Pal	Várzea da Palma
317100-Vazante	Vazante
317103-Verdelândia	Verdelândia
317107-Veredinha	Veredinha
317120-Vespasiano	Vespasiano
317130-Viçosa-Mg	Viçosa Mg
317170-Virgínia	Virgínia
317200-Visconde do R	Visconde do Rio Branco
320010-Afonso Cláudi	Afonso Cláudio
320013-Águia Branca	Águia Branca
320020-Alegre	Alegre
320030-Alfredo Chave	Alfredo Chaves
320040-Anchieta-Es	Anchieta Es
320060-Aracruz	Aracruz
320070-Atílio Vivacq	Atílio Vivácqua
320080-Baixo Guandu	Baixo Guandu
320090-Barra de São	Barra de São Francisco
320100-Boa Esperança	Boa Esperança do Iguaçu
320110-Bom Jesus do	Bom Jesus do
320115-Brejetuba	Brejetuba
320120-Cachoeiro de	Cachoeiro de Itapemirim
320130-Cariacica	Cariacica
320140-Castelo	Castelo
320150-Colatina	Colatina
320160-Conceição da	Conceição da Barra
320170-Conceição do	Conceição do Araguaia
320180-Divino de São	Divino de São Lourenço
320190-Domingos Mart	Domingos Martins
320200-Dores do Rio	Dores do Rio Preto
320210-Ecoporanga	Ecoporanga
320220-Fundão	Fundão
320230-Guaçuí	Guaçuí
320240-Guarapari	Guarapari
320245-Ibatiba	Ibatiba
320250-Ibiraçu	Ibiraçu
320255-Ibitirama	Ibitirama
320260-Iconha	Iconha
320270-Itaguaçu	Itaguaçu
320280-Itapemirim	Itapemirim
320290-Itarana	Itarana
320300-Iúna	Iúna
320305-Jaguaré	Jaguaré
320310-Jerônimo Mont	Jerônimo Monteiro
320313-João Neiva	João Neiva
320316-Laranja da Te	Laranja da Terra
320320-Linhares	Linhares
320332-Marataizes	Marataizes
320334-Marechal Flor	Marechal Flor
320335-Marilândia	Marilândia
320340-Mimoso do Sul	Mimoso do Sul
320350-Montanha	Montanha
320370-Muniz Freire	Muniz Freire
320390-Nova Venécia	Nova Venécia
320405-Pedro Canário	Pedro Canário
320410-Pinheiros	Pinheiros
320420-Piúma	Piúma
320430-Presidente Ke	Presidente Kennedy
320435-Rio Bananal	Rio Bananal
320440-Rio Novo do S	Rio Novo do S
320450-Santa Leopold	Santa Leopold
320455-Santa Maria d	Santa Maria da Vitória
320460-Santa Teresa	Santa Teresa
320465-São Domingos	São Domingos
320470-São Gabriel d	São Gabriel D
320480-São José do C	São José do Cedro
320490-São Mateus	São Mateus
320495-São Roque do	São Roque do Canaã
320500-Serra	Serra
320501-Sooretama	Sooretama
320503-Vargem Alta	Vargem Alta
320506-Venda Nova do	Venda Nova do Imigrante
320510-Viana-Es	Viana Es
320515-Vila Pavão	Vila Pavão
320520-Vila Velha	Vila Velha
320530-Vitória	Vitória
330010-Angra dos Rei	Angra dos Reis
330015-Aperibé	Aperibé
330020-Araruama	Araruama
330022-Areal	Areal
330023-Armação de Bú	Armação dos Búzios
330025-Arraial do Ca	Arraial do Cabo
330030-Barra do Pira	Barra do Pira
330040-Barra Mansa	Barra Mansa
330045-Belford Roxo	Belford Roxo
330050-Bom Jardim-Rj	Bom Jardim Rj
330060-Bom Jesus do	Bom Jesus do
330070-Cabo Frio	Cabo Frio
330080-Cachoeiras de	Cachoeiras de Macacu
330093-Carapebus	Carapebus
330095-Comendador Le	Comendador Levy Gasparian
330100-Campos dos Go	Campos dos Goytacazes
330110-Cantagalo-Rj	Cantagalo Rj
330120-Carmo	Carmo
330130-Casimiro de A	Casimiro de A
330140-Conceição de	Conceição de Macabu
330150-Cordeiro	Cordeiro
330160-Duas Barras	Duas Barras
330170-Duque de Caxi	Duque de Caxias
330180-Engenheiro Pa	Engenheiro Paulo de Frontin
330185-Guapimirim	Guapimirim
330187-Iguaba Grande	Iguaba Grande
330190-Itaboraí	Itaboraí
330200-Itaguaí	Itaguaí
330205-Italva	Italva
330220-Itaperuna	Itaperuna
330225-Itatiaia	Itatiaia
330227-Japeri	Japeri
330230-Laje do Muria	Laje do Muriaé
330240-Macaé	Macaé
330245-Macuco	Macuco
330250-Magé	Magé
330260-Mangaratiba	Mangaratiba
330270-Maricá	Maricá
330280-Mendes	Mendes
330285-Mesquita-Rj	Mesquita Rj
330290-Miguel Pereir	Miguel Pereira
330300-Miracema	Miracema
330310-Natividade-Rj	Natividade Rj
330320-Nilópolis	Nilópolis
330330-Niterói	Niterói
330340-Nova Friburgo	Nova Friburgo
330350-Nova Iguaçu	Nova Iguaçu
330360-Paracambi	Paracambi
330370-Paraíba do Su	Paraíba do Sul
330380-Parati	Parati
330385-Paty do Alfer	Paty do Alferes
330390-Petrópolis	Petrópolis
330395-Pinheiral	Pinheiral
330400-Piraí	Piraí
330410-Porciúncula	Porciúncula
330411-Porto Real	Porto Real
330412-Quatis	Quatis
330414-Queimados	Queimados
330415-Quissamã	Quissamã
330420-Resende	Resende
330430-Rio Bonito	Rio Bonito
330452-Rio das Ostra	Rio das Ostras
330455-Rio de Janeir	Rio de Janeiro
330460-Santa Maria M	Santa Maria Madalena
330470-Santo Antônio	Santo Antônio de Jesus
330475-São Francisco	São Francisco do Sul
330480-São Fidélis	São Fidélis
330490-São Gonçalo	São Gonçalo
330500-São João da B	São João da B
330510-São João de M	São João de Meriti
330515-São José do V	São José do Vale do Rio Preto
330520-São Pedro da	São Pedro da Água Branca
330540-Sapucaia-Rj	Sapucaia Rj
330550-Saquarema	Saquarema
330555-Seropédica	Seropédica
330560-Silva Jardim	Silva Jardim
330575-Tanguá	Tanguá
330580-Teresópolis	Teresópolis
330600-Três Rios	Três Rios
330610-Valença-Rj	Valença Rj
330615-Varre-Sai	Varre Sai
330620-Vassouras	Vassouras
330630-Volta Redonda	Volta Redonda
350010-Adamantina	Adamantina
350020-Adolfo	Adolfo
350030-Aguaí	Aguaí
350040-Águas da Prat	Águas da Prat
350050-Águas de Lind	Águas de Lindoia
350055-Águas de Sant	Águas de Sant
350060-Águas de São	Águas de São
350070-Agudos	Agudos
350075-Alambari	Alambari
350090-Altair	Altair
350100-Altinópolis	Altinópolis
350115-Alumínio	Alumínio
350120-Álvares Flore	Álvares Florence
350130-Álvares Macha	Álvares Machado
350160-Americana	Americana
350170-Américo Brasi	Américo Brasiliense
350190-Amparo-Sp	Amparo Sp
350200-Analândia	Analândia
350210-Andradina	Andradina
350220-Angatuba	Angatuba
350230-Anhembi	Anhembi
350240-Anhumas	Anhumas
350250-Aparecida-Sp	Aparecida Sp
350260-Aparecida D'O	Aparecida d'Oeste
350270-Apiaí	Apiaí
350275-Araçariguama	Araçariguama
350280-Araçatuba	Araçatuba
350290-Araçoiaba da	Araçoiaba da Serra
350300-Aramina	Aramina
350310-Arandu	Arandu
350320-Araraquara	Araraquara
350330-Araras	Araras
350340-Arealva	Arealva
350350-Areias	Areias
350360-Areiópolis	Areiópolis
350370-Ariranha	Ariranha
350380-Artur Nogueir	Artur Nogueira
350390-Arujá	Arujá
350400-Assis	Assis
350410-Atibaia	Atibaia
350420-Auriflama	Auriflama
350440-Avanhandava	Avanhandava
350450-Avaré	Avaré
350460-Bady Bassitt	Bady Bassitt
350480-Bálsamo	Bálsamo
350490-Bananal	Bananal
350510-Barbosa	Barbosa
350520-Bariri	Bariri
350530-Barra Bonita-	Barra Bonita
350550-Barretos	Barretos
350560-Barrinha	Barrinha
350570-Barueri	Barueri
350580-Bastos	Bastos
350590-Batatais	Batatais
350600-Bauru	Bauru
350610-Bebedouro	Bebedouro
350630-Bernardino de	Bernardino de Campos
350635-Bertioga	Bertioga
350640-Bilac	Bilac
350650-Birigui	Birigui
350660-Biritiba-Miri	Biritiba Miri
350670-Boa Esperança	Boa Esperança do Iguaçu
350680-Bocaina-Sp	Bocaina Sp
350690-Bofete	Bofete
350700-Boituva	Boituva
350710-Bom Jesus dos	Bom Jesus dos Perdões
350715-Bom Sucesso d	Bom Sucesso do Sul
350720-Borá	Borá
350730-Boracéia	Boracéia
350740-Borborema-Sp	Borborema Sp
350745-Borebi	Borebi
350750-Botucatu	Botucatu
350760-Bragança Paul	Bragança Paulista
350780-Brodósqui	Brodósqui
350790-Brotas	Brotas
350800-Buri	Buri
350810-Buritama	Buritama
350820-Buritizal	Buritizal
350830-Cabrália Paul	Cabrália Paul
350840-Cabreúva	Cabreúva
350850-Caçapava	Caçapava
350860-Cachoeira Pau	Cachoeira Pau
350880-Cafelândia-Sp	Cafelândia Sp
350890-Caiabu	Caiabu
350900-Caieiras	Caieiras
350920-Cajamar	Cajamar
350925-Cajati	Cajati
350930-Cajobi	Cajobi
350940-Cajuru	Cajuru
350945-Campina do Mo	Campina do Mo
350950-Campinas	Campinas
350960-Campo Limpo P	Campo Limpo Paulista
350970-Campos do Jor	Campos do Jordão
350980-Campos Novos	Campos Novos
350990-Cananéia	Cananéia
350995-Canas	Canas
351000-Cândido Mota	Cândido Mota
351015-Canitar	Canitar
351020-Capão Bonito	Capão Bonito
351030-Capela do Alt	Capela do Alto
351040-Capivari	Capivari
351050-Caraguatatuba	Caraguatatuba
351060-Carapicuíba	Carapicuíba
351080-Casa Branca	Casa Branca
351100-Castilho	Castilho
351110-Catanduva	Catanduva
351130-Cedral-Sp	Cedral Sp
351140-Cerqueira Cés	Cerqueira César
351150-Cerquilho	Cerquilho
351160-Cesário Lange	Cesário Lange
351170-Charqueada	Charqueada
351190-Clementina	Clementina
351200-Colina	Colina
351210-Colômbia	Colômbia
351220-Conchal	Conchal
351230-Conchas	Conchas
351240-Cordeirópolis	Cordeirópolis
351250-Coroados	Coroados
351260-Coronel Maced	Coronel Macedo
351270-Corumbataí	Corumbataí
351280-Cosmópolis	Cosmópolis
351290-Cosmorama	Cosmorama
351300-Cotia	Cotia
351310-Cravinhos	Cravinhos
351320-Cristais Paul	Cristais Paulista
351340-Cruzeiro	Cruzeiro
351350-Cubatão	Cubatão
351360-Cunha	Cunha
351370-Descalvado	Descalvado
351380-Diadema	Diadema
351385-Dirce Reis	Dirce Reis
351390-Divinolândia	Divinolândia
351400-Dobrada	Dobrada
351410-Dois Córregos	Dois Córregos
351430-Dourado	Dourado
351440-Dracena	Dracena
351450-Duartina	Duartina
351460-Dumont	Dumont
351470-Echaporã	Echaporã
351490-Elias Fausto	Elias Fausto
351492-Elisiário	Elisiário
351495-Embaúba	Embaúba
351500-Embu	Embu
351510-Embu-Guaçu	Embu Guaçu
351515-Engenheiro Co	Engenheiro Coelho
351518-Espírito Sant	Espírito Santo do Pinhal
351520-Estrela D'Oes	Estrela d'Oeste
351535-Euclides da C	Euclides da Cunha
351540-Fartura	Fartura
351550-Fernandópolis	Fernandópolis
351565-Fernão	Fernão
351570-Ferraz de Vas	Ferraz de Vas
351600-Flórida Pauli	Flórida Paulista
351610-Florínia	Florínia
351620-Franca	Franca
351630-Francisco Mor	Francisco Morato
351640-Franco da Roc	Franco da Rocha
351650-Gabriel Monte	Gabriel Monte
351660-Gália	Gália
351670-Garça	Garça
351685-Gavião Peixot	Gavião Peixoto
351690-General Salga	General Salga
351710-Glicério	Glicério
351720-Guaiçara	Guaiçara
351740-Guaíra-Sp	Guaíra Sp
351750-Guapiaçu	Guapiaçu
351760-Guapiara	Guapiara
351770-Guará	Guará
351780-Guaraçaí	Guaraçaí
351790-Guaraci-Sp	Guaraci Sp
351800-Guarani D'Oes	Guarani d'Oeste
351820-Guararapes	Guararapes
351830-Guararema	Guararema
351840-Guaratinguetá	Guaratinguetá
351850-Guareí	Guareí
351860-Guariba	Guariba
351870-Guarujá	Guarujá
351880-Guarulhos	Guarulhos
351885-Guatapará	Guatapará
351900-Herculândia	Herculândia
351905-Holambra	Holambra
351907-Hortolândia	Hortolândia
351910-Iacanga	Iacanga
351920-Iacri	Iacri
351930-Ibaté	Ibaté
351940-Ibirá	Ibirá
351950-Ibirarema	Ibirarema
351960-Ibitinga	Ibitinga
351970-Ibiúna	Ibiúna
351980-Icém	Icém
352000-Igaraçu do Ti	Igaraçu do Tietê
352010-Igarapava	Igarapava
352020-Igaratá	Igaratá
352030-Iguape	Iguape
352040-Ilhabela	Ilhabela
352042-Ilha Comprida	Ilha Comprida
352044-Ilha Solteira	Ilha Solteira
352050-Indaiatuba	Indaiatuba
352070-Indiaporã	Indiaporã
352090-Ipauçu	Ipauçu
352100-Iperó	Iperó
352110-Ipeúna	Ipeúna
352130-Ipuã	Ipuã
352140-Iracemápolis	Iracemápolis
352160-Irapuru	Irapuru
352170-Itaberá	Itaberá
352180-Itaí	Itaí
352190-Itajobi	Itajobi
352210-Itanhaém	Itanhaém
352215-Itaóca	Itaóca
352220-Itapecerica d	Itapecerica da Serra
352230-Itapetininga	Itapetininga
352240-Itapeva-Sp	Itapeva Sp
352250-Itapevi	Itapevi
352260-Itapira	Itapira
352270-Itápolis	Itápolis
352280-Itaporanga-Sp	Itaporanga Sp
352290-Itapuí	Itapuí
352310-Itaquaquecetu	Itaquaquecetuba
352320-Itararé	Itararé
352340-Itatiba	Itatiba
352350-Itatinga	Itatinga
352360-Itirapina	Itirapina
352380-Itobi	Itobi
352390-Itu	Itu
352400-Itupeva	Itupeva
352410-Ituverava	Ituverava
352430-Jaboticabal	Jaboticabal
352440-Jacareí	Jacareí
352450-Jaci	Jaci
352460-Jacupiranga	Jacupiranga
352470-Jaguariúna	Jaguariúna
352480-Jales	Jales
352490-Jambeiro	Jambeiro
352500-Jandira	Jandira
352510-Jardinópolis-	Jardinópolis
352520-Jarinu	Jarinu
352530-Jaú	Jaú
352540-Jeriquara	Jeriquara
352550-Joanópolis	Joanópolis
352570-José Bonifáci	José Bonifácio
352585-Jumirim	Jumirim
352590-Jundiaí	Jundiaí
352600-Junqueirópoli	Junqueirópolis
352610-Juquiá	Juquiá
352620-Juquitiba	Juquitiba
352630-Lagoinha	Lagoinha
352640-Laranjal Paul	Laranjal Paulista
352660-Lavrinhas	Lavrinhas
352670-Leme	Leme
352680-Lençóis Pauli	Lençóis Paulista
352690-Limeira	Limeira
352700-Lindóia	Lindóia
352710-Lins	Lins
352720-Lorena	Lorena
352730-Louveira	Louveira
352740-Lucélia	Lucélia
352760-Luís Antônio	Luís Antônio
352770-Luiziânia	Luiziânia
352780-Lupércio	Lupércio
352800-Macatuba	Macatuba
352810-Macaubal	Macaubal
352820-Macedônia	Macedônia
352830-Magda	Magda
352840-Mairinque	Mairinque
352850-Mairiporã	Mairiporã
352880-Maracaí	Maracaí
352885-Marapoama	Marapoama
352900-Marília	Marília
352920-Martinópolis	Martinópolis
352930-Matão	Matão
352940-Mauá	Mauá
352950-Mendonça	Mendonça
352970-Miguelópolis	Miguelópolis
352980-Mineiros do T	Mineiros do T
352990-Miracatu	Miracatu
353010-Mirandópolis	Mirandópolis
353020-Mirante do Pa	Mirante do Paranapanema
353030-Mirassol	Mirassol
353040-Mirassolândia	Mirassolândia
353050-Mococa	Mococa
353060-Moji das Cruz	Mogi das Cruzes
353070-Moji-Guaçu	Moji Guaçu
353080-Moji-Mirim	Moji Mirim
353090-Mombuca	Mombuca
353110-Mongaguá	Mongaguá
353120-Monte Alegre	Monte Alegre
353130-Monte Alto	Monte Alto
353140-Monte Aprazív	Monte Aprazív
353150-Monte Azul Pa	Monte Azul Paulista
353160-Monte Castelo	Monte Castelo
353180-Monte Mor	Monte Mor
353190-Morro Agudo	Morro Agudo
353200-Morungaba	Morungaba
353205-Motuca	Motuca
353220-Narandiba	Narandiba
353230-Natividade da	Natividade da Serra
353240-Nazaré Paulis	Nazaré Paulista
353250-Neves Paulist	Neves Paulista
353260-Nhandeara	Nhandeara
353280-Nova Aliança	Nova Aliança
353282-Nova Campina	Nova Campina
353290-Nova Europa	Nova Europa
353300-Nova Granada	Nova Granada
353320-Nova Independ	Nova Independ
353325-Novais	Novais
353340-Nova Odessa	Nova Odessa
353350-Novo Horizont	Novo Horizonte
353360-Nuporanga	Nuporanga
353390-Olímpia	Olímpia
353400-Onda Verde	Onda Verde
353410-Oriente	Oriente
353420-Orindiúva	Orindiúva
353430-Orlândia	Orlândia
353440-Osasco	Osasco
353460-Osvaldo Cruz	Osvaldo Cruz
353470-Ourinhos	Ourinhos
353475-Ouroeste	Ouroeste
353480-Ouro Verde-Sp	Ouro Verde Sp
353490-Pacaembu	Pacaembu
353500-Palestina-Sp	Palestina Sp
353510-Palmares Paul	Palmares Paul
353520-Palmeira D'Oe	Palmeira d'Oeste
353530-Palmital-Sp	Palmital Sp
353540-Panorama	Panorama
353550-Paraguaçu Pau	Paraguaçu Paulista
353560-Paraibuna	Paraibuna
353570-Paraíso-Sp	Paraíso Sp
353580-Paranapanema	Paranapanema
353600-Parapuã	Parapuã
353610-Pardinho	Pardinho
353620-Pariquera-Açu	Pariquera Açu
353630-Patrocínio Pa	Patrocínio Paulista
353640-Paulicéia	Paulicéia
353650-Paulínia	Paulínia
353670-Pederneiras	Pederneiras
353690-Pedranópolis	Pedranópolis
353700-Pedregulho	Pedregulho
353710-Pedreira	Pedreira
353715-Pedrinhas Pau	Pedrinhas Pau
353730-Penápolis	Penápolis
353740-Pereira Barre	Pereira Barre
353750-Pereiras	Pereiras
353760-Peruíbe	Peruíbe
353770-Piacatu	Piacatu
353780-Piedade	Piedade
353790-Pilar do Sul	Pilar do Sul
353800-Pindamonhanga	Pindamonhanga
353810-Pindorama	Pindorama
353820-Pinhalzinho-S	Pinhalzinho S
353830-Piquerobi	Piquerobi
353850-Piquete	Piquete
353860-Piracaia	Piracaia
353870-Piracicaba	Piracicaba
353880-Piraju	Piraju
353890-Pirajuí	Pirajuí
353900-Pirangi	Pirangi
353910-Pirapora do B	Pirapora do B
353920-Pirapozinho	Pirapozinho
353930-Pirassununga	Pirassununga
353950-Pitangueiras-	Pitangueiras
353960-Planalto-Sp	Planalto Sp
353970-Platina	Platina
353980-Poá	Poá
353990-Poloni	Poloni
354000-Pompéia	Pompéia
354020-Pontal	Pontal
354040-Populina	Populina
354050-Porangaba	Porangaba
354060-Porto Feliz	Porto Feliz
354070-Porto Ferreir	Porto Ferreira
354075-Potim	Potim
354080-Potirendaba	Potirendaba
354085-Pracinha	Pracinha
354090-Pradópolis	Pradópolis
354100-Praia Grande-	Praia Grande
354105-Pratânia	Pratânia
354120-Presidente Be	Presidente Bernardes
354130-Presidente Ep	Presidente Ep
354140-Presidente Pr	Presidente Prudente
354150-Presidente Ve	Presidente Venceslau
354160-Promissão	Promissão
354170-Quatá	Quatá
354180-Queiroz	Queiroz
354190-Queluz	Queluz
354200-Quintana	Quintana
354210-Rafard	Rafard
354220-Rancharia	Rancharia
354230-Redenção da S	Redenção da S
354240-Regente Feijó	Regente Feijó
354250-Reginópolis	Reginópolis
354260-Registro	Registro
354270-Restinga	Restinga
354290-Ribeirão Boni	Ribeirão Boni
354300-Ribeirão Bran	Ribeirão Bran
354310-Ribeirão Corr	Ribeirão Corrente
354320-Ribeirão do S	Ribeirão do S
354330-Ribeirão Pire	Ribeirão Pires
354340-Ribeirão Pret	Ribeirão Preto
354360-Rifaina	Rifaina
354370-Rincão	Rincão
354380-Rinópolis	Rinópolis
354390-Rio Claro-Sp	Rio Claro Sp
354400-Rio das Pedra	Rio das Pedras
354410-Rio Grande da	Rio Grande da Serra
354425-Rosana	Rosana
354430-Roseira	Roseira
354440-Rubiácea	Rubiácea
354450-Rubinéia	Rubinéia
354460-Sabino	Sabino
354480-Sales	Sales
354490-Sales Oliveir	Sales Oliveira
354500-Salesópolis	Salesópolis
354510-Salmourão	Salmourão
354515-Saltinho-Sp	Saltinho Sp
354520-Salto	Salto
354530-Salto de Pira	Salto de Pirapora
354540-Salto Grande	Salto Grande
354560-Santa Adélia	Santa Adélia
354570-Santa Alberti	Santa Alberti
354580-Santa Bárbara	Santa Bárbara d'Oeste
354600-Santa Branca	Santa Branca
354610-Santa Clara D	Santa Clara d'Oeste
354620-Santa Cruz da	Santa Cruz da Conceição
354630-Santa Cruz da	Santa Cruz da Conceição
354640-Santa Cruz do	Santa Cruz do Sul
354650-Santa Ernesti	Santa Ernestina
354660-Santa Fé do S	Santa Fé do Sul
354670-Santa Gertrud	Santa Gertrudes
354680-Santa Isabel-	Santa Isabel
354690-Santa Lúcia-S	Santa Lúcia S
354710-Santa Mercede	Santa Mercedes
354730-Santana de Pa	Santana de Parnaíba
354750-Santa Rita do	Santa Rita do Sapucaí
354760-Santa Rosa de	Santa Rosa de Viterbo
354765-Santa Salete	Santa Salete
354770-Santo Anastác	Santo Anastácio
354780-Santo André-S	Santo André S
354790-Santo Antônio	Santo Antônio de Jesus
354800-Santo Antônio	Santo Antônio de Jesus
354820-Santo Antônio	Santo Antônio de Jesus
354830-Santo Expedit	Santo Expedito do Sul
354840-Santópolis do	Santópolis do
354850-Santos	Santos
354860-São Bento do	São Bento do Sul
354870-São Bernardo	São Bernardo do Campo
354880-São Caetano d	São Caetano do Sul
354890-São Carlos-Sp	São Carlos Sp
354910-São João da B	São João da B
354940-São Joaquim d	São Joaquim da Barra
354970-São José do R	São José do Rio Preto
354980-São José do R	São José do Rio Preto
354990-São José dos	São José dos Campos
354995-São Lourenço	São Lourenço
355010-São Manuel	São Manuel
355020-São Miguel Ar	São Miguel Arcanjo
355030-São Paulo	São Paulo
355040-São Pedro-Sp	São Pedro Sp
355050-São Pedro do	São Pedro do Sul
355060-São Roque	São Roque
355070-São Sebastião	São Sebastião
355080-São Sebastião	São Sebastião
355090-São Simão-Sp	São Simão Sp
355100-São Vicente-S	São Vicente S
355110-Sarapuí	Sarapuí
355120-Sarutaiá	Sarutaiá
355130-Sebastianópol	Sebastianópolis do Sul
355140-Serra Azul	Serra Azul de Minas
355150-Serrana	Serrana
355160-Serra Negra	Serra Negra
355170-Sertãozinho-S	Sertãozinho S
355180-Sete Barras	Sete Barras
355210-Socorro	Socorro
355220-Sorocaba	Sorocaba
355230-Sud Mennucci	Sud Mennucci
355240-Sumaré	Sumaré
355250-Suzano	Suzano
355255-Suzanápolis	Suzanápolis
355260-Tabapuã	Tabapuã
355270-Tabatinga-Sp	Tabatinga Sp
355280-Taboão da Ser	Taboão da Serra
355290-Taciba	Taciba
355300-Taguaí	Taguaí
355310-Taiaçu	Taiaçu
355320-Taiúva	Taiúva
355330-Tambaú	Tambaú
355340-Tanabi	Tanabi
355350-Tapiraí-Sp	Tapiraí Sp
355360-Tapiratiba	Tapiratiba
355370-Taquaritinga	Taquaritinga
355380-Taquarituba	Taquarituba
355385-Taquarivaí	Taquarivaí
355390-Tarabaí	Tarabaí
355395-Tarumã	Tarumã
355400-Tatuí	Tatuí
355410-Taubaté	Taubaté
355430-Teodoro Sampa	Teodoro Sampaio
355440-Terra Roxa-Sp	Terra Roxa Sp
355450-Tietê	Tietê
355470-Torrinha	Torrinha
355480-Tremembé	Tremembé
355490-Três Fronteir	Três Fronteiras
355495-Tuiuti	Tuiuti
355500-Tupã	Tupã
355510-Tupi Paulista	Tupi Paulista
355535-Ubarana	Ubarana
355540-Ubatuba	Ubatuba
355560-Uchoa	Uchoa
355570-União Paulist	União Paulist
355580-Urânia	Urânia
355600-Urupês	Urupês
355610-Valentim Gent	Valentim Gentil
355620-Valinhos	Valinhos
355630-Valparaíso	Valparaíso
355635-Vargem-Sp	Vargem Sp
355640-Vargem Grande	Vargem Grande Paulista
355645-Vargem Grande	Vargem Grande Paulista
355650-Várzea Paulis	Várzea Paulista
355660-Vera Cruz-Sp	Vera Cruz Sp
355670-Vinhedo	Vinhedo
355680-Viradouro	Viradouro
355690-Vista Alegre	Vista Alegre
355700-Votorantim	Votorantim
355710-Votuporanga	Votuporanga
355720-Chavantes	Chavantes
355730-Estiva Gerbi	Estiva Gerbi
410020-Adrianópolis	Adrianópolis
410030-Agudos do Sul	Agudos do Sul
410040-Almirante Tam	Almirante Tamandaré
410060-Alto Paraná	Alto Paraná
410090-Amaporã	Amaporã
410100-Ampére	Ampére
410105-Anahy	Anahy
410110-Andirá	Andirá
410120-Antonina	Antonina
410140-Apucarana	Apucarana
410150-Arapongas	Arapongas
410160-Arapoti	Arapoti
410170-Araruna-Pr	Araruna Pr
410180-Araucária	Araucária
410190-Assaí	Assaí
410200-Assis Chateau	Assis Chateau
410210-Astorga	Astorga
410230-Balsa Nova	Balsa Nova
410240-Bandeirantes-	Bandeirantes
410250-Barbosa Ferra	Barbosa Ferraz
410280-Bela Vista do	Bela Vista do Paraíso
410290-Bituruna	Bituruna
410304-Boa Ventura d	Boa Ventura de São Roque
410310-Bocaiúva do S	Bocaiúva do Sul
410315-Bom Jesus do	Bom Jesus do
410320-Bom Sucesso-P	Bom Sucesso P
410322-Bom Sucesso d	Bom Sucesso do Sul
410335-Braganey	Braganey
410337-Brasilândia d	Brasilândia do Sul
410345-Cafelândia-Pr	Cafelândia Pr
410350-Califórnia	Califórnia
410360-Cambará	Cambará
410370-Cambé	Cambé
410380-Cambira	Cambira
410390-Campina da La	Campina da La
410400-Campina Grand	Campina Grande
410410-Campo do Tene	Campo do Tenente
410420-Campo Largo	Campo Largo
410425-Campo Magro	Campo Magro
410430-Campo Mourão	Campo Mourão
410442-Candói	Candói
410445-Cantagalo-Pr	Cantagalo Pr
410450-Capanema-Pr	Capanema Pr
410460-Capitão Leôni	Capitão Leônidas Marques
410465-Carambeí	Carambeí
410470-Carlópolis	Carlópolis
410480-Cascavel-Pr	Cascavel Pr
410490-Castro	Castro
410500-Catanduvas-Pr	Catanduvas Pr
410520-Cerro Azul	Cerro Azul
410530-Céu Azul	Céu Azul
410540-Chopinzinho	Chopinzinho
410550-Cianorte	Cianorte
410560-Cidade Gaúcha	Cidade Gaúcha
410570-Clevelândia	Clevelândia
410580-Colombo	Colombo
410590-Colorado-Pr	Colorado Pr
410600-Congonhinhas	Congonhinhas
410630-Corbélia	Corbélia
410640-Cornélio Proc	Cornélio Procópio
410645-Coronel Domin	Coronel Domin
410650-Coronel Vivid	Coronel Vivida
410660-Cruzeiro do O	Cruzeiro do Oeste
410670-Cruzeiro do S	Cruzeiro do Sul
410690-Curitiba	Curitiba
410700-Curiúva	Curiúva
410710-Diamante do N	Diamante do Norte
410720-Dois Vizinhos	Dois Vizinhos
410725-Douradina-Pr	Douradina Pr
410730-Doutor Camarg	Doutor Camargo
410750-Engenheiro Be	Engenheiro Be
410753-Entre Rios do	Entre Rios do Oeste
410760-Faxinal	Faxinal
410765-Fazenda Rio G	Fazenda Rio Grande
410780-Floraí	Floraí
410785-Flor da Serra	Flor da Serra
410790-Floresta-Pr	Floresta Pr
410800-Florestópolis	Florestópolis
410820-Formosa do Oe	Formosa do Oe
410830-Foz do Iguaçu	Foz do Iguaçu
410840-Francisco Bel	Francisco Bel
410845-Foz do Jordão	Foz do Jordão
410850-General Carne	General Carneiro
410860-Goioerê	Goioerê
410880-Guaíra-Pr	Guaíra Pr
410890-Guairaçá	Guairaçá
410930-Guaraniaçu	Guaraniaçu
410940-Guarapuava	Guarapuava
410960-Guaratuba	Guaratuba
410965-Honório Serpa	Honório Serpa
410970-Ibaiti	Ibaiti
410975-Ibema	Ibema
410980-Ibiporã	Ibiporã
411007-Imbaú	Imbaú
411010-Imbituva	Imbituva
411020-Inácio Martin	Inácio Martin
411040-Indianópolis-	Indianópolis
411050-Ipiranga	Ipiranga
411060-Iporã	Iporã
411070-Irati-Pr	Irati Pr
411080-Iretama	Iretama
411095-Itaipulândia	Itaipulândia
411100-Itambaracá	Itambaracá
411120-Itapejara D'O	Itapejara d'Oeste
411125-Itaperuçu	Itaperuçu
411140-Ivaí	Ivaí
411150-Ivaiporã	Ivaiporã
411155-Ivaté	Ivaté
411160-Ivatuba	Ivatuba
411180-Jacarezinho	Jacarezinho
411190-Jaguapitã	Jaguapitã
411200-Jaguariaíva	Jaguariaíva
411210-Jandaia do Su	Jandaia do Sul
411240-Japurá-Pr	Japurá Pr
411250-Jardim Alegre	Jardim Alegre
411260-Jardim Olinda	Jardim Olinda
411270-Jataizinho	Jataizinho
411275-Jesuítas	Jesuítas
411280-Joaquim Távor	Joaquim Távora
411290-Jundiaí do Su	Jundiaí do Su
411300-Jussara-Pr	Jussara Pr
411320-Lapa	Lapa
411330-Laranjeiras d	Laranjeiras do Sul
411345-Lindoeste	Lindoeste
411350-Loanda	Loanda
411360-Lobato	Lobato
411370-Londrina	Londrina
411375-Lunardelli	Lunardelli
411390-Mallet	Mallet
411400-Mamborê	Mamborê
411410-Mandaguaçu	Mandaguaçu
411420-Mandaguari	Mandaguari
411430-Mandirituba	Mandirituba
411435-Manfrinópolis	Manfrinópolis
411450-Manoel Ribas	Manoel Ribas
411460-Marechal Când	Marechal Cândido Rondon
411480-Marialva	Marialva
411490-Marilândia do	Marilândia do Sul
411520-Maringá	Maringá
411530-Mariópolis	Mariópolis
411535-Maripá	Maripá
411540-Marmeleiro	Marmeleiro
411560-Matelândia	Matelândia
411570-Matinhos	Matinhos
411580-Medianeira	Medianeira
411585-Mercedes	Mercedes
411590-Mirador-Pr	Mirador Pr
411605-Missal	Missal
411610-Moreira Sales	Moreira Sales
411620-Morretes	Morretes
411670-Nova Aurora-P	Nova Aurora P
411690-Nova Esperanç	Nova Esperanç
411695-Nova Esperanç	Nova Esperanç
411700-Nova Fátima-P	Nova Fátima P
411705-Nova Laranjei	Nova Laranjei
411710-Nova Londrina	Nova Londrina
411720-Nova Olímpia-	Nova Olímpia
411722-Nova Santa Ro	Nova Santa Ro
411725-Nova Prata do	Nova Prata do Iguaçu
411727-Nova Tebas	Nova Tebas
411730-Ortigueira	Ortigueira
411745-Ouro Verde do	Ouro Verde do Oeste
411750-Paiçandu	Paiçandu
411760-Palmas-Pr	Palmas Pr
411770-Palmeira-Pr	Palmeira Pr
411790-Palotina	Palotina
411800-Paraíso do No	Paraíso do Norte
411810-Paranacity	Paranacity
411820-Paranaguá	Paranaguá
411830-Paranapoema	Paranapoema
411840-Paranavaí	Paranavaí
411845-Pato Bragado	Pato Bragado
411850-Pato Branco	Pato Branco
411860-Paula Freitas	Paula Freitas
411880-Peabiru	Peabiru
411885-Perobal	Perobal
411890-Pérola	Pérola
411900-Pérola D'Oest	Pérola d'Oeste
411910-Piên	Piên
411915-Pinhais	Pinhais
411930-Pinhão-Pr	Pinhão Pr
411940-Piraí do Sul	Piraí do Sul
411950-Piraquara	Piraquara
411960-Pitanga	Pitanga
411970-Planaltina do	Planaltina do
411980-Planalto-Pr	Planalto Pr
411990-Ponta Grossa	Ponta Grossa
411995-Pontal do Par	Pontal do Par
412010-Porto Amazona	Porto Amazonas
412020-Porto Rico	Porto Rico
412030-Porto Vitória	Porto Vitória
412035-Pranchita	Pranchita
412050-Primeiro de M	Primeiro de Maio
412060-Prudentópolis	Prudentópolis
412065-Quarto Centen	Quarto Centenário
412070-Quatiguá	Quatiguá
412080-Quatro Barras	Quatro Barras
412085-Quatro Pontes	Quatro Pontes
412090-Quedas do Igu	Quedas do Iguaçu
412120-Quitandinha	Quitandinha
412140-Realeza	Realeza
412150-Rebouças	Rebouças
412160-Renascença	Renascença
412170-Reserva	Reserva
412175-Reserva do Ig	Reserva do Iguaçu
412180-Ribeirão Clar	Ribeirão Claro
412190-Ribeirão do P	Ribeirão do Pinhal
412200-Rio Azul	Rio Azul
412215-Rio Bonito do	Rio Bonito do Iguaçu
412220-Rio Branco do	Rio Branco do Sul
412230-Rio Negro-Pr	Rio Negro Pr
412240-Rolândia	Rolândia
412260-Rondon	Rondon
412270-Sabáudia	Sabáudia
412280-Salgado Filho	Salgado Filho
412300-Salto do Lont	Salto do Lontra
412310-Santa Amélia	Santa Amélia
412320-Santa Cecília	Santa Cecília
412330-Santa Cruz de	Santa Cruz de
412340-Santa Fé	Santa Fé do Araguaia
412350-Santa Helena-	Santa Helena
412370-Santa Isabel	Santa Isabel
412380-Santa Izabel	Santa Izabel
412390-Santa Mariana	Santa Mariana
412395-Santa Mônica	Santa Mônica
412402-Santa Tereza	Santa Tereza
412405-Santa Terezin	Santa Terezinha de Itaipu
412410-Santo Antônio	Santo Antônio de Jesus
412420-Santo Antônio	Santo Antônio de Jesus
412440-Santo Antônio	Santo Antônio de Jesus
412450-Santo Inácio	Santo Inácio
412460-São Carlos do	São Carlos do
412470-São Jerônimo	São Jerônimo da Serra
412480-São João-Pr	São João Pr
412500-São João do I	São João do Itaperiú
412510-São João do T	São João do Triunfo
412520-São Jorge D'O	São Jorge D'Oeste
412530-São Jorge do	São Jorge do Patrocínio
412535-São Jorge do	São Jorge do Patrocínio
412550-São José dos	São José dos Campos
412560-São Mateus do	São Mateus do Sul
412570-São Miguel do	São Miguel do Araguaia
412580-São Pedro do	São Pedro do Sul
412610-São Tomé-Pr	São Tomé Pr
412620-Sapopema	Sapopema
412625-Sarandi-Pr	Sarandi Pr
412630-Sengés	Sengés
412640-Sertaneja	Sertaneja
412660-Siqueira Camp	Siqueira Camp
412665-Sulina	Sulina
412667-Tamarana	Tamarana
412670-Tamboara	Tamboara
412680-Tapejara-Pr	Tapejara Pr
412700-Teixeira Soar	Teixeira Soares
412710-Telêmaco Borb	Telêmaco Borba
412720-Terra Boa	Terra Boa
412730-Terra Rica	Terra Rica
412740-Terra Roxa-Pr	Terra Roxa Pr
412750-Tibagi	Tibagi
412760-Tijucas do Su	Tijucas do Sul
412770-Toledo-Pr	Toledo Pr
412780-Tomazina	Tomazina
412785-Três Barras d	Três Barras do Paraná
412788-Tunas do Para	Tunas do Paraná
412790-Tuneiras do O	Tuneiras do O
412796-Turvo-Pr	Turvo Pr
412800-Ubiratã	Ubiratã
412810-Umuarama	Umuarama
412820-União da Vitó	União da Vitória
412840-Uraí	Uraí
412850-Wenceslau Bra	Wenceslau Bra
412853-Ventania	Ventania
412855-Vera Cruz do	Vera Cruz do
412860-Verê	Verê
412862-Vila Alta	Vila Alta
412870-Vitorino	Vitorino
420010-Abelardo Luz	Abelardo Luz
420020-Agrolândia	Agrolândia
420040-Água Doce	Água Doce
420055-Águas Frias	Águas Frias
420060-Águas Mornas	Águas Mornas
420075-Alto Bela Vis	Alto Bela Vista
420080-Anchieta-Sc	Anchieta Sc
420100-Anita Garibal	Anita Garibal
420110-Anitápolis	Anitápolis
420120-Antônio Carlo	Antônio Carlos
420125-Apiúna	Apiúna
420127-Arabutã	Arabutã
420130-Araquari	Araquari
420140-Araranguá	Araranguá
420150-Armazém	Armazém
420165-Arvoredo	Arvoredo
420170-Ascurra	Ascurra
420190-Aurora-Sc	Aurora Sc
420195-Balneário Arr	Balneário Arr
420200-Balneário Cam	Balneário Camboriú
420205-Balneário Bar	Balneário Bar
420207-Balneário Gai	Balneário Gai
420210-Barra Velha	Barra Velha
420220-Benedito Novo	Benedito Novo
420230-Biguaçu	Biguaçu
420240-Blumenau	Blumenau
420245-Bombinhas	Bombinhas
420250-Bom Jardim da	Bom Jardim da Serra
420253-Bom Jesus-Sc	Bom Jesus Sc
420260-Bom Retiro	Bom Retiro
420270-Botuverá	Botuverá
420280-Braço do Nort	Braço do Norte
420285-Braço do Trom	Braço do Trom
420290-Brusque	Brusque
420300-Caçador	Caçador
420310-Caibi	Caibi
420320-Camboriú	Camboriú
420330-Campo Alegre-	Campo Alegre
420340-Campo Belo do	Campo Belo do Sul
420350-Campo Erê	Campo Erê
420360-Campos Novos	Campos Novos
420370-Canelinha	Canelinha
420380-Canoinhas	Canoinhas
420390-Capinzal	Capinzal
420395-Capivari de B	Capivari de Baixo
420400-Catanduvas-Sc	Catanduvas Sc
420417-Cerro Negro	Cerro Negro
420420-Chapecó	Chapecó
420425-Cocal do Sul	Cocal do Sul
420430-Concórdia	Concórdia
420435-Cordilheira A	Cordilheira Alta
420440-Coronel Freit	Coronel Freit
420445-Coronel Marti	Coronel Martins
420450-Corupá	Corupá
420455-Correia Pinto	Correia Pinto
420460-Criciúma	Criciúma
420470-Cunha Porã	Cunha Porã
420480-Curitibanos	Curitibanos
420500-Dionísio Cerq	Dionísio Cerqueira
420515-Doutor Pedrin	Doutor Pedrinho
420520-Erval Velho	Erval Velho
420530-Faxinal dos G	Faxinal dos Guedes
420535-Flor do Sertã	Flor do Sertão
420540-Florianópolis	Florianópolis
420543-Formosa do Su	Formosa do Sul
420545-Forquilhinha	Forquilhinha
420550-Fraiburgo	Fraiburgo
420570-Garopaba	Garopaba
420580-Garuva	Garuva
420590-Gaspar	Gaspar
420600-Governador Ce	Governador Ce
420610-Grão Pará	Grão Pará
420620-Gravatal	Gravatal
420630-Guabiruba	Guabiruba
420640-Guaraciaba-Sc	Guaraciaba Sc
420650-Guaramirim	Guaramirim
420660-Guarujá do Su	Guarujá do Sul
420665-Guatambu	Guatambu
420670-Herval D'Oest	Herval d'Oeste
420675-Ibiam	Ibiam
420690-Ibirama	Ibirama
420700-Içara	Içara
420710-Ilhota	Ilhota
420720-Imaruí	Imaruí
420730-Imbituba	Imbituba
420740-Imbuia	Imbuia
420750-Indaial	Indaial
420757-Iomerê	Iomerê
420765-Iporã do Oest	Iporã do Oeste
420770-Ipumirim	Ipumirim
420775-Iraceminha	Iraceminha
420780-Irani	Irani
420800-Itá	Itá
420810-Itaiópolis	Itaiópolis
420820-Itajaí	Itajaí
420830-Itapema	Itapema
420840-Itapiranga-Sc	Itapiranga Sc
420845-Itapoá	Itapoá
420850-Ituporanga	Ituporanga
420860-Jaborá	Jaborá
420870-Jacinto Macha	Jacinto Machado
420880-Jaguaruna	Jaguaruna
420890-Jaraguá do Su	Jaraguá do Sul
420900-Joaçaba	Joaçaba
420910-Joinville	Joinville
420915-José Boiteux	José Boiteux
420917-Jupiá	Jupiá
420930-Lages	Lages
420940-Laguna	Laguna
420950-Laurentino	Laurentino
420960-Lauro Muller	Lauro Muller
420970-Lebon Régis	Lebon Régis
420985-Lindóia do Su	Lindóia do Sul
420990-Lontras	Lontras
421000-Luiz Alves	Luiz Alves
421003-Luzerna	Luzerna
421010-Mafra	Mafra
421020-Major Gercino	Major Gercino
421030-Major Vieira	Major Vieira
421040-Maracajá	Maracajá
421050-Maravilha-Sc	Maravilha Sc
421055-Marema	Marema
421060-Massaranduba-	Massaranduba
421080-Meleiro	Meleiro
421085-Mirim Doce	Mirim Doce
421090-Modelo	Modelo
421100-Mondaí	Mondaí
421105-Monte Carlo	Monte Carlo
421120-Morro da Fuma	Morro da Fumaça
421130-Navegantes	Navegantes
421140-Nova Erechim	Nova Erechim
421145-Nova Itaberab	Nova Itaberaba
421150-Nova Trento	Nova Trento
421160-Nova Veneza-S	Nova Veneza S
421165-Novo Horizont	Novo Horizonte
421170-Orleans	Orleans
421175-Otacílio Cost	Otacílio Costa
421190-Palhoça	Palhoça
421200-Palma Sola	Palma Sola
421205-Palmeira-Sc	Palmeira Sc
421210-Palmitos	Palmitos
421220-Papanduva	Papanduva
421223-Paraíso-Sc	Paraíso Sc
421227-Passos Maia	Passos Maia
421230-Paulo Lopes	Paulo Lopes
421240-Pedras Grande	Pedras Grande
421250-Penha	Penha
421260-Peritiba	Peritiba
421280-Piçarras	Balneário Piçarras
421290-Pinhalzinho-S	Pinhalzinho S
421300-Pinheiro Pret	Pinheiro Preto
421310-Piratuba	Piratuba
421315-Planalto Aleg	Planalto Aleg
421320-Pomerode	Pomerode
421340-Ponte Serrada	Ponte Serrada
421350-Porto Belo	Porto Belo
421360-Porto União	Porto União
421370-Pouso Redondo	Pouso Redondo
421380-Praia Grande-	Praia Grande
421400-Presidente Ge	Presidente Getúlio
421410-Presidente Ne	Presidente Ne
421415-Princesa	Princesa
421420-Quilombo	Quilombo
421440-Rio das Antas	Rio das Antas
421450-Rio do Campo	Rio do Campo
421470-Rio dos Cedro	Rio dos Cedros
421480-Rio do Sul	Rio do Sul
421490-Rio Fortuna	Rio Fortuna
421500-Rio Negrinho	Rio Negrinho
421505-Rio Rufino	Rio Rufino
421510-Rodeio	Rodeio
421530-Salete	Salete
421540-Salto Veloso	Salto Veloso
421545-Sangão	Sangão
421550-Santa Cecília	Santa Cecília
421560-Santa Rosa de	Santa Rosa de Viterbo
421565-Santa Rosa do	Santa Rosa do
421570-Santo Amaro d	Santo Amaro da Imperatriz
421580-São Bento do	São Bento do Sul
421600-São Carlos-Sc	São Carlos Sc
421605-São Cristóvão	São Cristóvão
421610-São Domingos-	São Domingos
421620-São Francisco	São Francisco do Sul
421625-São João do O	São João do Oeste
421630-São João Bati	São João Batista
421635-São João do I	São João do Itaperiú
421650-São Joaquim	São Joaquim
421660-São José	São José
421690-São Lourenço	São Lourenço
421700-São Ludgero	São Ludgero
421710-São Martinho-	São Martinho
421720-São Miguel D'	São Miguel do Iguaçu
421725-São Pedro de	São Pedro da Aldeia
421730-Saudades	Saudades
421740-Schroeder	Schroeder
421750-Seara	Seara
421755-Serra Alta	Serra Alta
421760-Siderópolis	Siderópolis
421770-Sombrio	Sombrio
421775-Sul Brasil	Sul Brasil
421780-Taió	Taió
421790-Tangará-Sc	Tangará Sc
421800-Tijucas	Tijucas
421810-Timbé do Sul	Timbé do Sul
421820-Timbó	Timbó
421825-Timbó Grande	Timbó Grande
421830-Três Barras	Três Barras
421835-Treviso	Treviso
421840-Treze de Maio	Treze de Maio
421850-Treze Tílias	Treze Tílias
421860-Trombudo Cent	Trombudo Cent
421870-Tubarão	Tubarão
421875-Tunápolis	Tunápolis
421880-Turvo-Sc	Turvo Sc
421885-União do Oest	União do Oeste
421890-Urubici	Urubici
421900-Urussanga	Urussanga
421910-Vargeão	Vargeão
421917-Vargem Bonita	Vargem Bonita
421920-Vidal Ramos	Vidal Ramos
421930-Videira	Videira
421940-Witmarsum	Witmarsum
421950-Xanxerê	Xanxerê
421970-Xaxim	Xaxim
430003-Aceguá	Aceguá
430010-Agudo	Agudo
430030-Alecrim	Alecrim
430040-Alegrete	Alegrete
430045-Alegria	Alegria
430047-Almirante Tam	Almirante Tamandaré
430057-Alto Feliz	Alto Feliz
430060-Alvorada-Rs	Alvorada Rs
430070-Anta Gorda	Anta Gorda
430080-Antônio Prado	Antônio Prado
430087-Araricá	Araricá
430090-Aratiba	Aratiba
430100-Arroio do Mei	Arroio do Meio
430105-Arroio do Sal	Arroio do Sal
430110-Arroio dos Ra	Arroio dos Ra
430120-Arroio do Tig	Arroio do Tigre
430130-Arroio Grande	Arroio Grande
430140-Arvorezinha	Arvorezinha
430150-Augusto Pesta	Augusto Pestana
430160-Bagé	Bagé
430163-Balneário Pin	Balneário Pin
430165-Barão	Barão
430180-Barracão-Rs	Barracão Rs
430190-Barra do Ribe	Barra do Ribeiro
430210-Bento Gonçalv	Bento Gonçalves
430223-Boa Vista do	Boa Vista do Sul
430230-Bom Jesus-Rs	Bom Jesus Rs
430235-Bom Princípio	Bom Princípio
430240-Bom Retiro do	Bom Retiro do Sul
430245-Boqueirão do	Boqueirão do Leão
430258-Bozano	Bozano
430265-Brochier	Brochier
430270-Butiá	Butiá
430280-Caçapava do S	Caçapava do Sul
430290-Cacequi	Cacequi
430300-Cachoeira do	Cachoeira do Sul
430310-Cachoeirinha-	Cachoeirinha
430340-Caiçara-Rs	Caiçara Rs
430350-Camaquã	Camaquã
430360-Cambará do Su	Cambará do Su
430370-Campina das M	Campina das M
430380-Campinas do S	Campinas do S
430390-Campo Bom	Campo Bom
430410-Campos Borges	Campos Borges
430420-Candelária	Candelária
430440-Canela	Canela
430450-Canguçu	Canguçu
430460-Canoas	Canoas
430462-Capão Bonito	Capão Bonito
430463-Capão da Cano	Capão da Cano
430466-Capão do Leão	Capão do Leão
430467-Capivari do S	Capivari do S
430468-Capela de San	Capela de Santana
430469-Capitão	Capitão
430470-Carazinho	Carazinho
430471-Caraã	Caraã
430480-Carlos Barbos	Carlos Barbosa
430490-Casca	Casca
430500-Catuípe	Catuípe
430510-Caxias do Sul	Caxias do Sul
430520-Cerro Largo	Cerro Largo
430530-Chapada	Chapada
430535-Charqueadas	Charqueadas
430543-Chuí	Chuí
430545-Cidreira	Cidreira
430570-Condor	Condor
430587-Coronel Barro	Coronel Barros
430590-Coronel Bicac	Coronel Bicaco
430593-Coronel Pilar	Coronel Pilar
430595-Cotiporã	Cotiporã
430597-Coxilha	Coxilha
430600-Crissiumal	Crissiumal
430605-Cristal	Cristal
430607-Cristal do Su	Cristal do Sul
430610-Cruz Alta	Cruz Alta
430620-Cruzeiro do S	Cruzeiro do Sul
430640-Dois Irmãos	Dois Irmãos
430645-Dois Lajeados	Dois Lajeados
430660-Dom Pedrito	Dom Pedrito
430670-Dona Francisc	Dona Francisca
430673-Doutor Mauríc	Doutor Maurício Cardoso
430676-Eldorado do S	Eldorado do Sul
430680-Encantado	Encantado
430690-Encruzilhada	Encruzilhada
430693-Entre-Ijuís	Entre Ijuís
430695-Entre Rios do	Entre Rios do Oeste
430700-Erechim	Erechim
430705-Ernestina	Ernestina
430730-Erval Seco	Erval Seco
430750-Espumoso	Espumoso
430755-Estação	Estação
430760-Estância Velh	Estância Velha
430770-Esteio	Esteio
430780-Estrela	Estrela
430790-Farroupilha	Farroupilha
430800-Faxinal do So	Faxinal do So
430807-Fazenda Vilan	Fazenda Vilan
430810-Feliz	Feliz
430820-Flores da Cun	Flores da Cunha
430850-Frederico Wes	Frederico Westphalen
430860-Garibaldi	Garibaldi
430880-General Câmar	General Câmara
430890-Getúlio Varga	Getúlio Vargas
430900-Giruá	Giruá
430905-Glorinha	Glorinha
430910-Gramado	Gramado
430920-Gravataí	Gravataí
430925-Guabiju	Guabiju
430930-Guaíba	Guaíba
430940-Guaporé	Guaporé
430955-Harmonia	Harmonia
430957-Herveiras	Herveiras
430960-Horizontina	Horizontina
430965-Hulha Negra	Hulha Negra
430990-Ibiraiaras	Ibiraiaras
430995-Ibirapuitã	Ibirapuitã
431000-Ibirubá	Ibirubá
431010-Igrejinha	Igrejinha
431020-Ijuí	Ijuí
431033-Imbé	Imbé
431036-Imigrante	Imigrante
431060-Itaqui	Itaqui
431080-Ivoti	Ivoti
431090-Jacutinga-Rs	Jacutinga Rs
431100-Jaguarão	Jaguarão
431110-Jaguari	Jaguari
431120-Júlio de Cast	Júlio de Cast
431130-Lagoa Vermelh	Lagoa Vermelha
431140-Lajeado-Rs	Lajeado Rs
431142-Lajeado do Bu	Lajeado do Bu
431162-Lindolfo Coll	Lindolfo Coll
431177-Maquiné	Maquiné
431179-Maratá	Maratá
431180-Marau	Marau
431215-Mato Leitão	Mato Leitão
431220-Maximiliano d	Maximiliano de Almeida
431225-Minas do Leão	Minas do Leão
431230-Miraguaí	Miraguaí
431240-Montenegro	Montenegro
431247-Morro Reuter	Morro Reuter
431260-Muçum	Muçum
431265-Não-Me-Toque	Não Me Toque
431267-Nicolau Vergu	Nicolau Vergueiro
431270-Nonoai	Nonoai
431280-Nova Araçá	Nova Araçá
431290-Nova Bassano	Nova Bassano
431300-Nova Bréscia	Nova Bréscia
431303-Nova Esperanç	Nova Esperanç
431306-Nova Hartz	Nova Hartz
431310-Nova Palma	Nova Palma
431320-Nova Petrópol	Nova Petrópolis
431330-Nova Prata	Nova Prata
431335-Nova Roma do	Nova Roma do Sul
431337-Nova Santa Ri	Nova Santa Rita
431339-Novo Cabrais	Novo Cabrais
431340-Novo Hamburgo	Novo Hamburgo
431350-Osório	Osório
431365-Palmares do S	Palmares do Sul
431370-Palmeira das	Palmeira das Missões
431390-Panambi	Panambi
431395-Pântano Grand	Pântano Grande
431400-Paraí	Paraí
431403-Pareci Novo	Pareci Novo
431405-Parobé	Parobé
431407-Passo do Sobr	Passo do Sobrado
431410-Passo Fundo	Passo Fundo
431413-Paulo Bento	Paulo Bento
431415-Paverama	Paverama
431420-Pedro Osório	Pedro Osório
431440-Pelotas	Pelotas
431442-Picada Café	Picada Café
431450-Pinheiro Mach	Pinheiro Mach
431460-Piratini	Piratini
431470-Planalto-Rs	Planalto Rs
431475-Poço das Anta	Poço das Antas
431480-Portão	Portão
431490-Porto Alegre	Porto Alegre
431513-Pouso Novo	Pouso Novo
431514-Presidente Lu	Presidente Lucena
431530-Quaraí	Quaraí
431550-Restinga Seca	Restinga Seca
431560-Rio Grande	Rio Grande
431570-Rio Pardo	Rio Pardo
431600-Rolante	Rolante
431640-Rosário do Su	Rosário do Su
431643-Saldanha Mari	Saldanha Marinho
431645-Salto do Jacu	Salto do Jacuí
431650-Salvador do S	Salvador do Sul
431660-Sananduva	Sananduva
431675-Santa Clara d	Santa Clara d'Oeste
431680-Santa Cruz do	Santa Cruz do Sul
431690-Santa Maria-R	Santa Maria R
431695-Santa Maria d	Santa Maria da Vitória
431710-Santana do Li	Santana do Livramento
431720-Santa Rosa	Santa Rosa
431730-Santa Vitória	Santa Vitória
431740-Santiago	Santiago
431750-Santo Ângelo	Santo Ângelo
431760-Santo Antônio	Santo Antônio de Jesus
431775-Santo Antônio	Santo Antônio de Jesus
431780-Santo Augusto	Santo Augusto
431790-Santo Cristo	Santo Cristo
431800-São Borja	São Borja
431805-São Domingos	São Domingos
431820-São Francisco	São Francisco do Sul
431830-São Gabriel-R	São Gabriel R
431840-São Jerônimo	São Jerônimo da Serra
431843-São João do P	São João do P
431844-São Jorge	São Jorge D'Oeste
431848-São José do H	São José do Herval
431849-São José do I	São José do Inhacorá
431850-São José do N	São José do Norte
431860-São José do O	São José do Ouro
431861-São José do S	São José do Sabugi
431862-São José dos	São José dos Campos
431870-São Leopoldo	São Leopoldo
431880-São Lourenço	São Lourenço
431890-São Luiz Gonz	São Luiz Gonzaga
431900-São Marcos	São Marcos
431910-São Martinho-	São Martinho
431935-São Pedro da	São Pedro da Água Branca
431937-São Pedro do	São Pedro do Sul
431950-São Sebastião	São Sebastião
431960-São Sepé	São Sepé
431975-São Vendelino	São Vendelino
431980-São Vicente d	São Vicente de Minas
431990-Sapiranga	Sapiranga
432000-Sapucaia do S	Sapucaia do Sul
432010-Sarandi-Rs	Sarandi Rs
432020-Seberi	Seberi
432026-Segredo	Segredo
432030-Selbach	Selbach
432035-Sentinela do	Sentinela do
432040-Serafina Corr	Serafina Corrêa
432050-Sertão	Sertão
432055-Sertão Santan	Sertão Santan
432067-Sinimbu	Sinimbu
432080-Soledade-Rs	Soledade Rs
432090-Tapejara-Rs	Tapejara Rs
432100-Tapera	Tapera
432120-Taquara	Taquara
432130-Taquari	Taquari
432140-Tenente Porte	Tenente Portela
432143-Terra de Arei	Terra de Areia
432145-Teutônia	Teutônia
432147-Tiradentes do	Tiradentes do Sul
432150-Torres	Torres
432160-Tramandaí	Tramandaí
432162-Travesseiro	Travesseiro
432163-Três Arroios	Três Arroios
432166-Três Cachoeir	Três Cachoeiras
432170-Três Coroas	Três Coroas
432180-Três de Maio	Três de Maio
432185-Três Palmeira	Três Palmeiras
432190-Três Passos	Três Passos
432195-Trindade do S	Trindade do Sul
432200-Triunfo-Rs	Triunfo Rs
432225-Tupandi	Tupandi
432230-Tuparendi	Tuparendi
432240-Uruguaiana	Uruguaiana
432250-Vacaria	Vacaria
432253-Vale do Sol	Vale do Sol
432254-Vale Real	Vale Real
432260-Venâncio Aire	Venâncio Aires
432270-Vera Cruz-Rs	Vera Cruz Rs
432280-Veranópolis	Veranópolis
432285-Vespasiano Co	Vespasiano Corrêa
432290-Viadutos	Viadutos
432300-Viamão	Viamão
432330-Vila Flores	Vila Flores
432340-Vila Maria	Vila Maria
432377-Westfália	Westfália
432380-Xangri-Lá	Xangri Lá
500020-Água Clara	Água Clara
500060-Amambaí	Amambaí
500070-Anastácio	Anastácio
500080-Anaurilândia	Anaurilândia
500085-Angélica	Angélica
500090-Antônio João	Antônio João
500100-Aparecida do	Aparecida do Taboado
500110-Aquidauana	Aquidauana
500124-Aral Moreira	Aral Moreira
500190-Bataguassu	Bataguassu
500200-Bataiporã	Bataiporã
500210-Bela Vista	Bela Vista
500215-Bodoquena	Bodoquena
500220-Bonito-Ms	Bonito Ms
500230-Brasilândia	Brasilândia
500240-Caarapó	Caarapó
500260-Camapuã	Camapuã
500270-Campo Grande-	Campo Grande
500290-Cassilândia	Cassilândia
500295-Chapadão do S	Chapadão do Sul
500320-Corumbá	Corumbá
500325-Costa Rica	Costa Rica
500330-Coxim	Coxim
500370-Dourados	Dourados
500375-Eldorado-Ms	Eldorado Ms
500380-Fátima do Sul	Fátima do Sul
500410-Guia Lopes da	Guia Lopes da Laguna
500440-Inocência	Inocência
500450-Itaporã	Itaporã
500460-Itaquiraí	Itaquiraí
500470-Ivinhema	Ivinhema
500490-Jaraguari	Jaraguari
500500-Jardim-Ms	Jardim Ms
500520-Ladário	Ladário
500540-Maracaju	Maracaju
500560-Miranda	Miranda
500568-Mundo Novo-Ms	Mundo Novo Ms
500570-Naviraí	Naviraí
500580-Nioaque	Nioaque
500600-Nova Alvorada	Nova Alvorada do Sul
500620-Nova Andradin	Nova Andradin
500625-Novo Horizont	Novo Horizonte
500630-Paranaíba	Paranaíba
500640-Pedro Gomes	Pedro Gomes
500660-Ponta Porã	Ponta Porã
500710-Ribas do Rio	Ribas do Rio Pardo
500720-Rio Brilhante	Rio Brilhante
500740-Rio Verde de	Rio Verde de Mato Grosso
500750-Rochedo	Rochedo
500755-Santa Rita do	Santa Rita do Sapucaí
500769-São Gabriel d	São Gabriel D
500790-Sidrolândia	Sidrolândia
500793-Sonora	Sonora
500795-Tacuru	Tacuru
500800-Terenos	Terenos
500830-Três Lagoas	Três Lagoas
510020-Água Boa-Mt	Água Boa Mt
510025-Alta Floresta	Alta Floresta
510035-Alto Boa Vist	Alto Boa Vista
510040-Alto Garças	Alto Garças
510060-Alto Taquari	Alto Taquari
510080-Apiacás	Apiacás
510125-Araputanga	Araputanga
510130-Arenápolis	Arenápolis
510140-Aripuanã	Aripuanã
510170-Barra do Bugr	Barra do Bugres
510180-Barra do Garç	Barra do Garças
510185-Bom Jesus do	Bom Jesus do
510190-Brasnorte	Brasnorte
510250-Cáceres	Cáceres
510263-Campo Novo do	Campo Novo do Parecis
510267-Campo Verde	Campo Verde
510268-Campos de Júl	Campos de Júlio
510269-Canabrava do	Canabrava do
510270-Canarana-Mt	Canarana Mt
510300-Chapada dos G	Chapada dos Guimarães
510310-Cocalinho	Cocalinho
510320-Colíder	Colíder
510325-Colniza	Colniza
510330-Comodoro	Comodoro
510335-Confresa	Confresa
510337-Cotriguaçu	Cotriguaçu
510340-Cuiabá	Cuiabá
510345-Denise	Denise
510350-Diamantino	Diamantino
510360-Dom Aquino	Dom Aquino
510385-Gaúcha do Nor	Gaúcha do Norte
510410-Guarantã do N	Guarantã do Norte
510480-Jaciara	Jaciara
510490-Jangada	Jangada
510500-Jauru	Jauru
510510-Juara	Juara
510515-Juína	Juína
510517-Juruena	Juruena
510520-Juscimeira	Juscimeira
510525-Lucas do Rio	Lucas do Rio Verde
510550-Vila Bela da	Vila Bela da Santíssima Trindade
510558-Marcelândia	Marcelândia
510560-Matupá	Matupá
510562-Mirassol D'Oe	Mirassol d'Oeste
510590-Nobres	Nobres
510600-Nortelândia	Nortelândia
510610-Nossa Senhora	Nossa Senhora
510615-Nova Bandeira	Nova Bandeira
510618-Nova Lacerda	Nova Lacerda
510621-Nova Canaã do	Nova Canaã do Norte
510622-Nova Mutum	Nova Mutum
510623-Nova Olímpia-	Nova Olímpia
510624-Nova Ubiratã	Nova Ubiratã
510625-Nova Xavantin	Nova Xavantina
510626-Novo Mundo	Novo Mundo
510628-Novo São Joaq	Novo São Joaquim
510629-Paranaíta	Paranaíta
510630-Paranatinga	Paranatinga
510637-Pedra Preta-M	Pedra Preta M
510642-Peixoto de Az	Peixoto de Az
510645-Planalto da S	Planalto da Serra
510650-Poconé	Poconé
510665-Pontal do Ara	Pontal do Ara
510675-Pontes e Lace	Pontes e Lacerda
510677-Porto Alegre	Porto Alegre
510680-Porto dos Gaú	Porto dos Gaúchos
510700-Poxoréo	Poxoréo
510704-Primavera do	Primavera do Leste
510706-Querência	Querência
510710-São José dos	São José dos Campos
510730-São José do R	São José do Rio Preto
510735-São José do X	São José do Xingu
510760-Rondonópolis	Rondonópolis
510770-Rosário Oeste	Rosário Oeste
510776-Santa Rita do	Santa Rita do Sapucaí
510779-Santo Antônio	Santo Antônio de Jesus
510780-Santo Antônio	Santo Antônio de Jesus
510785-São Félix do	São Félix do Xingu
510787-Sapezal	Sapezal
510788-Serra Nova Do	Serra Nova Dourada
510790-Sinop	Sinop
510792-Sorriso	Sorriso
510794-Tabaporã	Tabaporã
510795-Tangará da Se	Tangará da Serra
510800-Tapurah	Tapurah
510805-Terra Nova do	Terra Nova do Norte
510840-Várzea Grande	Várzea Grande
510850-Vera	Vera
510860-Vila Rica	Vila Rica
510885-Nova Marilând	Nova Marilândia
510890-Nova Maringá	Nova Maringá
520005-Abadia de Goi	Abadia de Goi
520010-Abadiânia	Abadiânia
520013-Acreúna	Acreúna
520025-Águas Lindas	Águas Lindas
520030-Alexânia	Alexânia
520055-Alto Horizont	Alto Horizont
520110-Anápolis	Anápolis
520130-Anicuns	Anicuns
520140-Aparecida de	Aparecida de Goiânia
520150-Aporé	Aporé
520180-Aragoiânia	Aragoiânia
520250-Aruanã	Aruanã
520280-Avelinópolis	Avelinópolis
520320-Barro Alto-Go	Barro Alto Go
520330-Bela Vista de	Bela Vista de Goiás
520350-Bom Jesus de	Bom Jesus de Goiás
520380-Britânia	Britânia
520390-Buriti Alegre	Buriti Alegre
520400-Cabeceiras	Cabeceiras
520410-Cachoeira Alt	Cachoeira Alt
520425-Cachoeira Dou	Cachoeira Dou
520430-Caçu	Caçu
520450-Caldas Novas	Caldas Novas
520485-Campo Limpo d	Campo Limpo de Goiás
520500-Carmo do Rio	Carmo do Rio Claro
520510-Catalão	Catalão
520520-Caturaí	Caturaí
520530-Cavalcante	Cavalcante
520540-Ceres	Ceres
520545-Cezarina	Cezarina
520547-Chapadão do C	Chapadão do Céu
520549-Cidade Ociden	Cidade Ocidental
520551-Cocalzinho de	Cocalzinho de Goiás
520570-Córrego do Ou	Córrego do Ouro
520590-Corumbaíba	Corumbaíba
520620-Cristalina	Cristalina
520640-Crixás	Crixás
520725-Doverlândia	Doverlândia
520740-Edéia	Edéia
520750-Estrela do No	Estrela do Norte
520790-Flores de Goi	Flores de Goiás
520800-Formosa	Formosa
520840-Goianápolis	Goianápolis
520860-Goianésia	Goianésia
520870-Goiânia	Goiânia
520880-Goianira	Goianira
520890-Goiás	Goiás
520910-Goiatuba	Goiatuba
520970-Hidrolândia-G	Hidrolândia G
520995-Indiara	Indiara
521000-Inhumas	Inhumas
521010-Ipameri	Ipameri
521020-Iporá	Iporá
521040-Itaberaí	Itaberaí
521090-Itapaci	Itapaci
521140-Itauçu	Itauçu
521150-Itumbiara	Itumbiara
521170-Jandaia	Jandaia
521180-Jaraguá	Jaraguá
521190-Jataí	Jataí
521210-Joviânia	Joviânia
521220-Jussara-Go	Jussara Go
521230-Leopoldo de B	Leopoldo de Bulhões
521250-Luziânia	Luziânia
521270-Mambaí	Mambaí
521300-Maurilândia	Maurilândia
521308-Minaçu	Minaçu
521310-Mineiros	Mineiros
521370-Montes Claros	Montes Claros
521375-Montividiu	Montividiu
521380-Morrinhos-Go	Morrinhos Go
521380-Morrinhos-Go	Morrinhos Go
521400-Mozarlândia	Mozarlândia
521400-Mozarlândia	Mozarlândia
521440-Nazário	Nazário
521450-Nerópolis	Nerópolis
521460-Niquelândia	Niquelândia
521500-Nova Veneza-G	Nova Veneza G
521523-Novo Gama	Novo Gama
521530-Orizona	Orizona
521540-Ouro Verde de	Ouro Verde de
521550-Ouvidor	Ouvidor
521560-Padre Bernard	Padre Bernard
521570-Palmeiras de	Palmeiras de Goiás
521630-Paranaiguara	Paranaiguara
521640-Paraúna	Paraúna
521645-Perolândia	Perolândia
521680-Petrolina de	Petrolina de Goiás
521710-Piracanjuba	Piracanjuba
521730-Pirenópolis	Pirenópolis
521740-Pires do Rio	Pires do Rio
521760-Planaltina	Planaltina
521770-Pontalina	Pontalina
521800-Porangatu	Porangatu
521830-Posse	Posse
521839-Professor Jam	Professor Jamil
521850-Quirinópolis	Quirinópolis
521860-Rialma	Rialma
521878-Rio Quente	Rio Quente
521880-Rio Verde	Rio Verde
521890-Rubiataba	Rubiataba
521910-Santa Bárbara	Santa Bárbara d'Oeste
521925-Santa Fé de G	Santa Fé de Goiás
521930-Santa Helena	Santa Helena
521945-Santa Rita do	Santa Rita do Sapucaí
521970-Santa Terezin	Santa Terezinha de Itaipu
521971-Santo Antônio	Santo Antônio de Jesus
521973-Santo Antônio	Santo Antônio de Jesus
521975-Santo Antônio	Santo Antônio de Jesus
521980-São Domingos-	São Domingos
522010-São Luís de M	São Luís de Montes Belos
522020-São Miguel do	São Miguel do Araguaia
522026-São Miguel do	São Miguel do Araguaia
522040-São Simão-Go	São Simão Go
522045-Senador Caned	Senador Canedo
522050-Serranópolis	Serranópolis
522060-Silvânia	Silvânia
522068-Simolândia	Simolândia
522140-Trindade-Go	Trindade Go
522155-Turvelândia	Turvelândia
522160-Uruaçu	Uruaçu
522170-Uruana	Uruana
522180-Urutaí	Urutaí
522185-Valparaíso de	Valparaíso de Goiás
522200-Vianópolis	Vianópolis
522205-Vicentinópoli	Vicentinópoli
522220-Vila Boa	Vila Boa
522230-Vila Propício	Vila Propício
530010-Brasília	Brasília"""

    
    mapping = {}
    lines = mapping_data.strip().split('\n')
    for line in lines:
        parts = line.split('\t')
        if len(parts) == 2:
            mapping[parts[0]] = parts[1]
    
    # Salva no arquivo
    with open("municipio_mapping.txt", "w", encoding="utf-8") as f:
        for key, value in mapping.items():
            f.write(f"{key}\t{value}\n")
    
    print(f"✅ Mapeamento de municípios criado com {len(mapping)} entradas")
    return mapping

def load_municipio_mapping() -> Dict[str, str]:
    """Carrega o mapeamento de município do arquivo gerado."""
    mapping = {}
    try:
        # Tenta carregar o arquivo gerado
        with open("municipio_mapping.txt", "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) == 2:
                    mapping[parts[0]] = parts[1]
        print(f"✅ Mapeamento de municípios carregado: {len(mapping)} entradas")
    except FileNotFoundError:
        st.warning("Arquivo 'municipio_mapping.txt' não encontrado. Criando mapeamento...")
        mapping = create_municipio_mapping_file()
    return mapping

# --------------- Mapeamento DINÂMICO ---------------
def detect_and_map_columns(df: pd.DataFrame) -> dict:
    """
    Detecta automaticamente as colunas baseado nos nomes normalizados
    e retorna um mapeamento para os nomes padrão
    """
    col_mapping = {}
    # Normaliza os nomes das colunas do DataFrame para facilitar a comparação
    normalized_cols = {normalize_name(col): col for col in df.columns}
    
    # Mapeamento de padrões para nossas colunas padrão - EXPANDIDO
    patterns = {
        'data': ['data_acidente', 'data', 'dt_acidente', 'dataacidente', 'data_acidente_1', 'datadoacidente'],
        'uf': ['uf_munic_acidente', 'uf', 'uf_municipio', 'uf_acidente', 'uf_munic_empregador', 'ufmunicempregador'],
        'setor': ['cnae2_0_empregador_1', 'setor', 'cnae_descricao', 'atividade', 'empregador', 'cnae20empregador1'],
        'cnae_codigo': ['cnae2_0_empregador', 'cnae', 'cnae_codigo', 'codigo_cnae', 'cnae20empregador'],
        'lesao': ['natureza_da_lesao', 'lesao', 'natureza_lesao', 'tipo_lesao', 'naturezalesao'],
        'origem': ['agente_causador_acidente', 'origem', 'agente_causador', 'causa', 'agentecausadoracidente'],
        'tipo_acidente': ['tipo_do_acidente', 'tipo_acidente', 'acidente_tipo', 'tipodoacidente'],
        'municipio': ['municipio', 'munic', 'municipio_acidente', 'munic_empr', 'municempr'],
        'munic_empr': ['munic_empr', 'municipio_empregador', 'municempregador', 'munic_empregador'],
        'uf_munic_empregador': ['uf_munic_empregador', 'ufempregador', 'uf_municipio_empregador', 'ufmunicempregador']
    }
    
    # Para debug: mostrar colunas normalizadas disponíveis
    st.write("🔍 Colunas normalizadas disponíveis:", list(normalized_cols.keys()))
    
    # Busca por correspondências
    for standard_name, possible_names in patterns.items():
        found = False
        # Primeiro, verifica correspondências exatas
        for norm_name in possible_names:
            if norm_name in normalized_cols:
                col_mapping[standard_name] = normalized_cols[norm_name]
                st.write(f"✅ Mapeado: {standard_name} ← {normalized_cols[norm_name]} (correspondência exata)")
                found = True
                break
        
        # Se não encontrou correspondência exata, busca por padrões parciais
        if not found:
            for original_col, normalized_col in normalized_cols.items():
                for pattern in possible_names:
                    if pattern in normalized_col:
                        col_mapping[standard_name] = original_col
                        st.write(f"🔄 Mapeado: {standard_name} ← {original_col} (correspondência parcial: '{pattern}' em '{normalized_col}')")
                        found = True
                        break
                if found:
                    break
    
    return col_mapping

def apply_uf_and_municipio_mapping(df: pd.DataFrame) -> pd.DataFrame:
    """Aplica os mapeamentos de UF e Município do Empregador."""
    
    # Mapeamento de UF (UF Munic. Empregador -> Novo_Nome_UF Munic. Empregador)
    if 'uf_munic_empregador' in df.columns:
        st.write("Aplicando mapeamento de UF...")
        
        # Primeiro, normalizar os dados de UF
        df['uf_munic_empregador_normalized'] = df['uf_munic_empregador'].astype(str).apply(
            lambda x: _strip_accents(x).strip().title() if pd.notna(x) else x
        )
        
        # Aplicar o mapeamento
        df['uf_empregador_sigla'] = df['uf_munic_empregador_normalized'].map(UF_MAPPING)
        
        # Para debug: mostrar valores únicos antes e depois
        st.write("Valores únicos em uf_munic_empregador:", df['uf_munic_empregador'].unique()[:10])
        st.write("Valores únicos em uf_munic_empregador_normalized:", df['uf_munic_empregador_normalized'].unique()[:10])
        st.write("Valores únicos em uf_empregador_sigla:", df['uf_empregador_sigla'].unique()[:10])
        
        # Contar quantos foram mapeados
        total_uf = len(df)
        mapeados_uf = df['uf_empregador_sigla'].notna().sum()
        st.write(f"UFs mapeadas: {mapeados_uf}/{total_uf} ({mapeados_uf/total_uf*100:.1f}%)")
        
        # Preencher os não mapeados com o valor original
        df['uf_empregador_sigla'] = df['uf_empregador_sigla'].fillna(df['uf_munic_empregador'])
    
    # Mapeamento de Município (Munic Empr -> Municipio_Novo)
    municipio_mapping = load_municipio_mapping()
    if municipio_mapping:
        # Tenta diferentes nomes de coluna para município
        municipio_cols = ['munic_empr', 'municipio_empregador', 'munic_empregador']
        municipio_col_found = None
        
        for col in municipio_cols:
            if col in df.columns:
                municipio_col_found = col
                break
        
        if municipio_col_found:
            st.write(f"Aplicando mapeamento de município na coluna: {municipio_col_found}")
            
            # Normalizar a coluna de município antes do mapeamento
            df['municipio_empregador_novo'] = df[municipio_col_found].map(municipio_mapping)
            
            # Conta quantos foram mapeados
            total_munic = len(df)
            mapeados_munic = df['municipio_empregador_novo'].notna().sum()
            st.write(f"Municípios mapeados: {mapeados_munic}/{total_munic} ({mapeados_munic/total_munic*100:.1f}%)")
            
            # Preenche os não mapeados com o valor original
            df['municipio_empregador_novo'] = df['municipio_empregador_novo'].fillna(df[municipio_col_found])
        else:
            st.error("❌ Nenhuma coluna de município do empregador encontrada")
            st.write("Colunas disponíveis:", [col for col in df.columns if 'munic' in col.lower()])
    else:
        st.error("❌ Mapeamento de municípios não carregado")
        
    return df

def check_uf_mapping(df: pd.DataFrame):
    """Verifica e mostra estatísticas do mapeamento de UF"""
    if 'uf_munic_empregador' in df.columns and 'uf_empregador_sigla' in df.columns:
        st.subheader("🔍 Verificação do Mapeamento de UF")
        
        # Mostrar valores únicos e contagens
        uf_counts = df['uf_munic_empregador'].value_counts().head(20)
        uf_sigla_counts = df['uf_empregador_sigla'].value_counts().head(20)
        
        col1, col2 = st.columns(2)
        with col1:
            st.write("Top 20 UFs Originais:")
            st.dataframe(uf_counts)
        with col2:
            st.write("Top 20 UFs Mapeadas:")
            st.dataframe(uf_sigla_counts)
        
        # Verificar especificamente o Paraná
        parana_original = df[df['uf_munic_empregador'].str.contains('paran', case=False, na=False)]
        parana_mapeado = df[df['uf_empregador_sigla'] == 'PR']
        
        st.write(f"Registros com 'Paran' no nome original: {len(parana_original)}")
        st.write(f"Registros mapeados como PR: {len(parana_mapeado)}")
        
        if len(parana_original) > 0 and len(parana_mapeado) > 0:
            st.success("✅ Paraná detectado e mapeado corretamente!")
        elif len(parana_original) > 0 and len(parana_mapeado) == 0:
            st.error("❌ Paraná detectado mas NÃO mapeado!")
            st.write("Valores originais do Paraná:", parana_original['uf_munic_empregador'].unique())

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
st.title("Observatório — CSV (Agregação de Múltiplos Arquivos)")

# --- CORREÇÃO 1: Usar st.session_state para manter o DataFrame em cache ---
if 'df_full' not in st.session_state:
    st.session_state.df_full = None
if 'df_filtered' not in st.session_state:
    st.session_state.df_filtered = None

with st.sidebar:
    st.header("Fonte dos Dados")
    mode = st.radio("Como fornecer os arquivos?", ["Pasta com múltiplos CSVs", "Carregar arquivo único"], index=1, key="load_mode")
    
    folder_path = None
    upload = None

    if mode == "Pasta com múltiplos CSVs":
        folder_path = st.text_input("Caminho da pasta com CSVs", value="C:/Users/Gabriel/Documents/CSVs", key="folder_path")
        st.info("💡 Coloque todos os CSVs na mesma pasta")
    else:
        upload = st.file_uploader("Envie seu .csv", type=["csv"], key="file_uploader")

    with st.expander("Opções avançadas (se precisar)"):
        sep_label = st.selectbox("Delimitador", ["Automático", ";", ",", "\\t", "|"], index=0, key="sep_label")
        sep_opt = None if sep_label == "Automático" else ("\t" if sep_label == "\\t" else sep_label)
        decimal_opt = st.selectbox("Separador decimal", [",", "."], index=0, key="decimal_opt")
        skiprows = st.number_input("Pular linhas iniciais", min_value=0, max_value=500, value=0, step=1, key="skiprows")
        enc_first = st.selectbox("Encoding preferido", ["latin1 (BR)", "utf-8-sig", "utf-8", "cp1252"], index=0, key="enc_first")
        enc_order = [enc_first.split(" ")[0]] + [e for e in ENCODINGS_BR if e != enc_first.split(" ")[0]]
    
    # Botão para criar/carregar mapeamento
    if st.button("🔄 Criar/Carregar Mapeamento de Municípios"):
        create_municipio_mapping_file()
        st.success("Mapeamento de municípios criado/carregado com sucesso!")
    
    # --- CORREÇÃO 2: Usar um callback para limpar o cache se o botão for clicado ---
    def clear_cache():
        st.session_state.df_full = None
        st.session_state.df_filtered = None

    run = st.button("Carregar dados", on_click=clear_cache)

# --------------- Carregar & preparar ---------------
if st.session_state.df_full is None:
    if not run and upload is None:
        st.info("👈 Selecione a pasta com CSVs ou carregue um arquivo, depois clique em **Carregar dados**.")
        st.stop()

    try:
        if mode == "Pasta com múltiplos CSVs":
            if not folder_path or not os.path.exists(folder_path):
                st.error(f"Pasta não encontrada: {folder_path}")
                st.stop()
            
            df_raw = load_all_csvs_from_folder(
                folder_path, sep_opt=sep_opt, decimal_opt=decimal_opt,
                skiprows=skiprows, encodings=enc_order
            )
            st.success(f"✅ Dados agregados carregados com sucesso! Total: {df_raw.shape[0]:,} registros")
            
        else:  # Modo arquivo único
            if upload is None:
                st.error("Selecione um arquivo CSV para carregar.")
                st.stop()
            
            df_raw, enc_used, sep_used = load_csv_simple(
                upload, sep_opt=sep_opt, decimal_opt=decimal_opt,
                skiprows=skiprows, encodings=enc_order
            )
            st.success(f"CSV carregado. **Encoding:** {enc_used} | **Separador:** {repr(sep_used)} | Linhas: {df_raw.shape[0]:,} | Colunas: {df_raw.shape[1]}")
            
    except Exception as e:
        st.error(f"Não consegui ler os dados: {e}")
        st.stop()

    # normalização + remoção de duplicadas
    df = normalize_headers(df_raw)
    df = ensure_unique_columns(df)
    
    # Mostrar colunas disponíveis para debug
    with st.expander("🔍 Colunas disponíveis (após normalização)"):
        st.write("Colunas encontradas:", list(df.columns))
    
    # Detectar e mapear colunas automaticamente
    col_mapping = detect_and_map_columns(df)
    
    st.subheader("🔧 Mapeamento de Colunas Detectado")
    st.write("O sistema detectou automaticamente estas correspondências:")
    for std_name, orig_name in col_mapping.items():
        st.write(f"• **{std_name}** ← {orig_name}")
    
    # Verificar se temos as colunas mínimas necessárias
    required_cols = ['data', 'uf', 'setor', 'lesao', 'origem', 'tipo_acidente']
    missing_required = [col for col in required_cols if col not in col_mapping]
    
    if missing_required:
        st.warning(f"⚠️ Algumas colunas importantes não foram detectadas: {missing_required}")
        st.info("Vou tentar usar colunas alternativas...")
    
    # Aplicar o mapeamento - renomear apenas as colunas detectadas
    df_renamed = df.copy()
    for std_name, orig_name in col_mapping.items():
        df_renamed[std_name] = df[orig_name]
    
    # limpa espaços
    for c in df_renamed.select_dtypes(include=['object']).columns:
        df_renamed[c] = df_renamed[c].astype(str).str.strip()
    
    # datas derivadas
    ensure_datetime(df_renamed, "data")
    if "data" in df_renamed:
        df_renamed["ano"] = df_renamed["data"].dt.year
        df_renamed["mes"] = df_renamed["data"].dt.to_period("M").astype(str)
    
    # uf/região
    if "uf" in df_renamed:
        df_renamed["uf_sigla"] = df_renamed["uf"].apply(derive_sigla_from_name)
        df_renamed["regiao"] = df_renamed["uf_sigla"].apply(derive_regiao_from_sigla)
        
    # Aplica os mapeamentos de UF e Município do Empregador
    df_renamed = apply_uf_and_municipio_mapping(df_renamed)

    # VERIFICAÇÃO ESPECÍFICA DO MAPEAMENTO
    check_uf_mapping(df_renamed)

    # Debug: mostrar primeiras linhas das colunas mapeadas
    st.subheader("🔍 Debug - Verificação do Mapeamento")
    if 'munic_empr' in df_renamed.columns:
        st.write("Amostra de dados de Munic Empr (original):")
        st.write(df_renamed['munic_empr'].head(10))
    if 'municipio_empregador_novo' in df_renamed.columns:
        st.write("Amostra de dados de municipio_empregador_novo (mapeado):")
        st.write(df_renamed['municipio_empregador_novo'].head(10))
    if 'uf_munic_empregador' in df_renamed.columns:
        st.write("Amostra de dados de UF Munic. Empregador (original):")
        st.write(df_renamed['uf_munic_empregador'].head(10))
    if 'uf_empregador_sigla' in df_renamed.columns:
        st.write("Amostra de dados de uf_empregador_sigla (mapeado):")
        st.write(df_renamed['uf_empregador_sigla'].head(10))

    # Armazena o DataFrame completo processado no session_state
    st.session_state.df_full = df_renamed
    st.session_state.df_filtered = df_renamed.copy()

# Se o DataFrame já estiver em cache, recupera
df_renamed = st.session_state.df_full

# --------------- Filtros globais ---------------
st.header("Filtros globais")
df_f = df_renamed.copy()

# Criar colunas para os filtros
col1, col2, col3, col4, col5, col6 = st.columns(6)

# Filtro por UF (se disponível)
if "uf_sigla" in df_f:
    ufs = sorted([u for u in df_f["uf_sigla"].dropna().unique().tolist() if u])
    uf_sel = col1.multiselect("UF (sigla)", ufs, default=[], key="uf_filter")
    if uf_sel:
        df_f = df_f[df_f["uf_sigla"].isin(uf_sel)]

# Filtro por Região (se disponível)
if "regiao" in df_f:
    regioes = sorted([r for r in df_f["regiao"].dropna().unique().tolist() if r])
    reg_sel = col2.multiselect("Região", regioes, default=[], key="regiao_filter")
    if reg_sel:
        df_f = df_f[df_f["regiao"].isin(reg_sel)]

# Filtro por Mês (se disponível)
if "mes" in df_f:
    meses = sorted(df_f["mes"].dropna().unique().tolist())
    mes_sel = col3.selectbox("Mês (YYYY-MM)", ["(todos)"] + meses, index=0, key="mes_filter")
    if mes_sel != "(todos)":
        df_f = df_f[df_f["mes"] == mes_sel]

# Filtro por Ano (se disponível)
if "ano" in df_f:
    anos = sorted(df_f["ano"].dropna().unique().tolist())
    ano_sel = col4.selectbox("Ano", ["(todos)"] + anos, index=0, key="ano_filter")
    if ano_sel != "(todos)":
        df_f = df_f[df_f["ano"] == ano_sel]

# Filtro por Tipo de Acidente (se disponível)
if "tipo_acidente" in df_f:
    tipo_opts = sorted(df_f["tipo_acidente"].dropna().astype(str).unique().tolist())
    tipo_sel = col5.multiselect("Tipo de acidente", tipo_opts, default=[], key="tipo_acidente_filter")
    if tipo_sel:
        df_f = df_f[df_f["tipo_acidente"].astype(str).isin(tipo_sel)]

# Filtro por CNAE (código) (se disponível)
if "cnae_codigo" in df_f:
    cnae_codigos = sorted([c for c in df_f["cnae_codigo"].dropna().unique().tolist() if c])
    cnae_sel = col6.multiselect("CNAE (código)", cnae_codigos, default=[], key="cnae_codigo_filter")
    if cnae_sel:
        df_f = df_f[df_f["cnae_codigo"].astype(str).isin(cnae_sel)]

# Filtro adicional por descrição do CNAE (se disponível)
if "setor" in df_f:
    cnae_descricoes = sorted([d for d in df_f["setor"].dropna().unique().tolist() if d])
    cnae_desc_sel = st.multiselect("CNAE (setor/atividade)", cnae_descricoes, default=[], key="cnae_desc_filter")
    if cnae_desc_sel:
        df_f = df_f[df_f["setor"].astype(str).isin(cnae_desc_sel)]

# Filtro por arquivo de origem (se aplicável)
if 'arquivo_origem' in df_f.columns:
    arquivos = sorted(df_f['arquivo_origem'].unique().tolist())
    arquivo_sel = st.multiselect("Filtrar por arquivo de origem", arquivos, default=[], key="arquivo_origem_filter")
    if arquivo_sel:
        df_f = df_f[df_f['arquivo_origem'].isin(arquivo_sel)]

with st.expander("🔎 Filtro por termo (texto livre)"):
    termo = st.text_input("Digite um termo para filtrar (procura em colunas de texto). Deixe vazio para ignorar.", key="termo_filter")
    if termo:
        termo_lower = termo.lower()
        text_cols = [c for c in df_f.columns if df_f[c].dtype == "object"]
        mask = pd.Series(False, index=df_f.index)
        for c in text_cols:
            mask = mask | df_f[c].astype(str).str.lower().str.contains(termo_lower, na=False)
        df_f = df_f[mask]

# Atualiza o DataFrame filtrado no session_state
st.session_state.df_filtered = df_f

# --------------- Abas / Dashboards ---------------
df_display = st.session_state.df_filtered

tab_names = ["📊 Visão geral", "⏱ Série temporal", "🗺️ UF/Região", "🏭 Setor/CNAE", "🩹 Tipo de Lesão", "⚙️ Origem/Causa", "📋 Dados + Download"]

# Filtrar abas baseado nas colunas disponíveis
available_tabs = []
if any(col in df_display.columns for col in ['mes', 'ano', 'uf_sigla', 'setor', 'lesao', 'origem', 'tipo_acidente']):
    available_tabs = tab_names
else:
    available_tabs = ["📋 Dados + Download"]

tabs = st.tabs(available_tabs)

# Visão geral
if "📊 Visão geral" in available_tabs:
    with tabs[available_tabs.index("📊 Visão geral")]:
        st.subheader("Visão geral (dados filtrados)")
        k1, k2, k3, k4, k5 = st.columns(5)
        total = df_display.shape[0]
        with k1: st.metric("Registros", f"{total:,}")
        
        if 'arquivo_origem' in df_display.columns:
            with k2: st.metric("Arquivos", f"{df_display['arquivo_origem'].nunique():,}")

        if "mes" in df_display and df_display["mes"].notna().any():
            serie = df_display.groupby("mes").size().sort_index()
            if len(serie) > 0:
                ultimo = int(serie.iloc[-1])
                delta = int(ultimo - (serie.iloc[-2] if len(serie) > 1 else 0))
                with k3: st.metric("Último mês (qtd.)", f"{ultimo:,}", delta=f"{delta:+,}")
            else:
                with k3: st.metric("Último mês (qtd.)", "—")
        else:
            with k3: st.metric("Último mês (qtd.)", "—")

        if "uf_sigla" in df_display:
            with k4: st.metric("UFs cobertas", f"{df_display['uf_sigla'].nunique():,}")
        else:
            with k4: st.metric("UFs cobertas", "—")

        if "setor" in df_display:
            with k5: st.metric("Setores/Atividades", f"{df_display['setor'].nunique():,}")
        else:
            with k5: st.metric("Setores/Atividades", "—")

        st.markdown("---")
        cA, cB = st.columns([2, 1])
        with cA:
            if "mes" in df_display and df_display["mes"].notna().any():
                st.caption("Registros por mês")
                monthly_data = df_display.groupby("mes").size().sort_index()
                if len(monthly_data) > 0:
                    st.line_chart(monthly_data)
                else:
                    st.info("Não há dados para mostrar o gráfico mensal.")
            else:
                st.info("Não há coluna de mês derivada.")
        with cB:
            top_n = st.number_input("Top N (rankings)", min_value=5, max_value=50, value=10, step=1, key="top_n_geral")
            if "uf_sigla" in df_display:
                st.caption(f"Top {top_n} — UF")
                uf_counts = df_display["uf_sigla"].value_counts().head(top_n)
                if len(uf_counts) > 0:
                    st.bar_chart(uf_counts)
            if "setor" in df_display:
                st.caption(f"Top {top_n} — Setor/Atividade")
                setor_counts = df_display["setor"].astype(str).value_counts().head(top_n)
                if len(setor_counts) > 0:
                    st.bar_chart(setor_counts)

# Série temporal
if "⏱ Série temporal" in available_tabs:
    with tabs[available_tabs.index("⏱ Série temporal")]:
        st.subheader("Evolução Temporal dos Acidentes")
        
        if "data" in df_display:
            # Agrupamento por ano/mês para a série temporal
            df_time = df_display.copy()
            df_time['ano_mes'] = df_time['data'].dt.to_period('M')
            
            # Contagem de acidentes por ano/mês
            monthly_counts = df_time.groupby('ano_mes').size().rename('Total de Acidentes')
            monthly_counts.index = monthly_counts.index.astype(str)
            
            st.caption("Total de Acidentes por Mês")
            st.line_chart(monthly_counts)
            
            # Agrupamento por ano para consolidação anual
            df_time['ano'] = df_time['data'].dt.year
            yearly_counts = df_time.groupby('ano').size().rename('Total de Acidentes')
            
            st.caption("Total de Acidentes por Ano (Consolidação)")
            st.bar_chart(yearly_counts)
            
            # Tabela de dados
            st.markdown("---")
            st.caption("Dados Mensais")
            st.dataframe(monthly_counts.reset_index(), use_container_width=True)
            st.caption("Dados Anuais")
            st.dataframe(yearly_counts.reset_index(), use_container_width=True)
            
        else:
            st.warning("A coluna de data ('data') não foi mapeada corretamente ou está ausente no dataset filtrado.")

# UF/Região
if "🗺️ UF/Região" in available_tabs:
    with tabs[available_tabs.index("🗺️ UF/Região")]:
        st.subheader("Distribuição por UF e Região")
        
        if "regiao" in df_display:
            st.caption("Acidentes por Região")
            regiao_counts = df_display["regiao"].value_counts()
            st.bar_chart(regiao_counts)
            
            st.caption("Acidentes por UF")
            uf_counts = df_display["uf_sigla"].value_counts()
            st.bar_chart(uf_counts)
            
            st.markdown("---")
            st.caption("Tabela de Distribuição")
            
            # Tabela de contagem por UF e Região
            uf_regiao_counts = df_display.groupby(['regiao', 'uf_sigla']).size().reset_index(name='Total de Acidentes')
            st.dataframe(uf_regiao_counts, use_container_width=True)
        else:
            st.warning("As colunas 'uf_sigla' ou 'regiao' não foram mapeadas corretamente ou estão ausentes.")

# Setor/CNAE
if "🏭 Setor/CNAE" in available_tabs:
    with tabs[available_tabs.index("🏭 Setor/CNAE")]:
        st.subheader("Distribuição por Setor de Atividade (CNAE)")
        
        if "setor" in df_display:
            top_n_setor = st.number_input("Top N Setores", min_value=5, max_value=50, value=15, step=1, key="top_n_setor")
            
            setor_counts = df_display["setor"].astype(str).value_counts().head(top_n_setor)
            
            st.caption(f"Top {top_n_setor} Setores/Atividades com mais acidentes")
            st.bar_chart(setor_counts)
            
            st.markdown("---")
            st.caption("Tabela de Contagem por Setor (Top 50)")
            st.dataframe(df_display["setor"].astype(str).value_counts().head(50), use_container_width=True)
        else:
            st.warning("A coluna 'setor' não foi mapeada corretamente ou está ausente.")

# Tipo de Lesão
if "🩹 Tipo de Lesão" in available_tabs:
    with tabs[available_tabs.index("🩹 Tipo de Lesão")]:
        st.subheader("Distribuição por Tipo de Lesão")
        
        if "lesao" in df_display:
            lesao_counts = df_display["lesao"].value_counts()
            
            st.caption("Contagem de acidentes por Natureza da Lesão")
            st.bar_chart(lesao_counts)
            
            st.markdown("---")
            st.caption("Tabela de Contagem por Lesão")
            st.dataframe(lesao_counts, use_container_width=True)
        else:
            st.warning("A coluna 'lesao' não foi mapeada corretamente ou está ausente.")

# Origem/Causa
if "⚙️ Origem/Causa" in available_tabs:
    with tabs[available_tabs.index("⚙️ Origem/Causa")]:
        st.subheader("Distribuição por Agente Causador (Origem)")
        
        if "origem" in df_display:
            top_n_origem = st.number_input("Top N Agentes Causadores", min_value=5, max_value=50, value=15, step=1, key="top_n_origem")
            
            origem_counts = df_display["origem"].value_counts().head(top_n_origem)
            
            st.caption(f"Top {top_n_origem} Agentes Causadores de Acidentes")
            st.bar_chart(origem_counts)
            
            st.markdown("---")
            st.caption("Tabela de Contagem por Agente Causador (Top 50)")
            st.dataframe(df_display["origem"].value_counts().head(50), use_container_width=True)
        else:
            st.warning("A coluna 'origem' não foi mapeada corretamente ou está ausente.")

# Dados + Download
if "📋 Dados + Download" in available_tabs:
    with tabs[available_tabs.index("📋 Dados + Download")]:
        st.subheader("Dados brutos (após filtros)")
        st.write(f"Mostrando {df_display.shape[0]:,} registros.")
        st.dataframe(df_display, use_container_width=True)
        
        # Opções de download
        col1, col2 = st.columns(2)
        with col1:
            st.download_button("⬇️ Baixar CSV filtrado",
                            data=df_display.to_csv(index=False).encode("utf-8-sig"),
                            file_name="dados_filtrados.csv", mime="text/csv")
        with col2:
            st.download_button("⬇️ Baixar dados completos (todos os CSVs)",
                            data=df_renamed.to_csv(index=False).encode("utf-8-sig"),
                            file_name="dados_completos_agregados.csv", mime="text/csv")

# Perfil opcional
with st.expander("🧭 Perfil do dataset"):
    n_rows, n_cols = df_renamed.shape
    st.caption(f"**Linhas:** {n_rows:,} | **Colunas:** {n_cols}")
    profile = pd.DataFrame({
        "coluna": df_renamed.columns,
        "dtype": [str(t) for t in df_renamed.dtypes.values],
        "n_nulos": [int(df_renamed[c].isna().sum()) for c in df_renamed.columns],
        "%_nulos": [round(df_renamed[c].isna().mean()*100, 2) for c in df_renamed.columns],
        "n_unicos": [int(df_renamed[c].nunique(dropna=True)) for c in df_renamed.columns],
    })
    st.dataframe(profile, use_container_width=True)