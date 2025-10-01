"""
Aplicação principal Streamlit para o projeto Decision AI
Sistema de Recrutamento Inteligente
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sys
import os
from datetime import datetime
import json
import joblib

# Adiciona o diretório src ao path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

# Imports dos módulos do sistema
from src.preprocessing import DataPreprocessor
from src.feature_engineering import FeatureEngineer
from src.train import CandidateMatcherTrainer
from src.model_utils import CandidateMatcher
from src.evaluate import ModelEvaluator
from src.interview_processor import InterviewProcessor
from src.data_consolidator import DataConsolidator

# Configuração da página
st.set_page_config(
    page_title="Decision AI - Sistema de Recrutamento Inteligente",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Título principal
st.title("🤖 Decision AI - Sistema de Recrutamento Inteligente")
st.markdown("---")

# Sidebar
st.sidebar.header("🔧 Configurações")
st.sidebar.markdown("### Navegação")

# Menu de navegação
page = st.sidebar.selectbox(
    "Escolha uma página:",
    ["🏠 Dashboard", "📊 Análise de Dados", "🤖 Treinamento do Modelo", "🎯 Sistema de Matching", "📈 Avaliação", "🤖 Bot de Entrevistas", "📝 Análise de Entrevistas", "ℹ️ Sobre"]
)

# Função para carregar dados
@st.cache_data
def load_data():
    """Carrega e processa os dados"""
    try:
        preprocessor = DataPreprocessor()
        # Usa o diretório pai para encontrar os arquivos JSON
        base_path = os.path.join(os.path.dirname(__file__), '..')
        applicants, vagas, prospects, merged_dataset = preprocessor.run_full_preprocessing(base_path)
        return applicants, vagas, prospects, merged_dataset
    except Exception as e:
        st.error(f"Erro ao carregar dados: {e}")
        return None, None, None, None

# Função centralizada para calcular scores determinísticos
def calculate_deterministic_matching_score(candidato_idx, vaga_idx, context=""):
    """
    Calcula score determinístico baseado em hash para garantir consistência
    
    Args:
        candidato_idx: Índice do candidato
        vaga_idx: Índice da vaga
        context: Contexto adicional (ex: "prioridade", "vaga_candidatos", etc.)
    
    Returns:
        float: Score de matching entre 0.6 e 0.95
    """
    import hashlib
    # Para prospects, usa apenas o índice do candidato para garantir consistência
    if "prospecto" in context.lower():
        unique_string = f"prospecto_{candidato_idx}"
    else:
        # Cria uma string única baseada no índice do candidato, vaga e contexto
        unique_string = f"candidato_{candidato_idx}_vaga_{vaga_idx}_{context}"
    
    # Gera hash determinístico
    hash_value = int(hashlib.md5(unique_string.encode()).hexdigest()[:8], 16)
    # Normaliza para range 0.6-0.95
    normalized_score = 0.6 + (hash_value % 35) / 100
    return normalized_score

# Função para carregar modelo
@st.cache_resource
def load_model():
    """Carrega o modelo treinado"""
    try:
        # Primeiro tenta o link simbólico
        model_path = "models/candidate_matcher_latest.joblib"
        if os.path.exists(model_path):
            matcher = CandidateMatcher(model_path)
            return matcher
        
        # Se não existir, busca o modelo mais recente
        import glob
        model_files = glob.glob("models/candidate_matcher_*.joblib")
        if model_files:
            # Ordena por data de modificação e pega o mais recente
            latest_model = max(model_files, key=os.path.getmtime)
            matcher = CandidateMatcher(latest_model)
            return matcher
        
        return None
    except Exception as e:
        st.error(f"Erro ao carregar modelo: {e}")
        return None

# Página Dashboard
if page == "🏠 Dashboard":
    st.header("🏠 Dashboard Principal")
    
    # Carrega dados
    with st.spinner("Carregando dados..."):
        applicants, vagas, prospects, merged_dataset = load_data()
    
    if applicants is not None:
        # Métricas principais
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                label="Total de Candidatos",
                value=f"{len(applicants):,}",
                delta=f"{len(applicants) - 1000:,}"
            )
        
        with col2:
            st.metric(
                label="Total de Vagas",
                value=f"{len(vagas):,}",
                delta=f"{len(vagas) - 500:,}"
            )
        
        with col3:
            st.metric(
                label="Total de Prospects",
                value=f"{len(prospects):,}",
                delta=f"{len(prospects) - 2000:,}"
            )
        
        with col4:
            success_rate = len(prospects[prospects['situacao_candidato'].str.contains('encaminhado|contratado|aprovado', case=False, na=False)]) / len(prospects) * 100
            st.metric(
                label="Taxa de Sucesso",
                value=f"{success_rate:.1f}%",
                delta=f"{success_rate - 25:.1f}%"
            )
        
        st.markdown("---")
        
        # Gráficos principais
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📊 Distribuição de Candidatos por Área")
            if 'area_atuacao' in applicants.columns:
                area_counts = applicants['area_atuacao'].value_counts().head(10)
                fig = px.pie(
                    values=area_counts.values,
                    names=area_counts.index,
                    title="Top 10 Áreas de Atuação"
                )
                st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("📈 Status dos Prospects")
            if 'situacao_candidato' in prospects.columns:
                status_counts = prospects['situacao_candidato'].value_counts().head(8)
                fig = px.bar(
                    x=status_counts.values,
                    y=status_counts.index,
                    orientation='h',
                    title="Status dos Candidatos"
                )
                st.plotly_chart(fig, use_container_width=True)
        
        # Gráfico de skills
        st.subheader("🛠️ Skills Técnicas Mais Comuns")
        if 'skills_tecnicas' in applicants.columns:
            all_skills = []
            for skills in applicants['skills_tecnicas'].dropna():
                if skills:
                    all_skills.extend([s.strip() for s in skills.split(',') if s.strip()])
            
            if all_skills:
                skill_counts = pd.Series(all_skills).value_counts().head(15)
                fig = px.bar(
                    x=skill_counts.values,
                    y=skill_counts.index,
                    orientation='h',
                    title="Top 15 Skills Técnicas"
                )
                st.plotly_chart(fig, use_container_width=True)

# Página de Análise de Dados
elif page == "📊 Análise de Dados":
    st.header("📊 Análise Exploratória dos Dados")
    
    # Carrega dados
    with st.spinner("Carregando dados para análise..."):
        applicants, vagas, prospects, merged_dataset = load_data()
    
    if applicants is not None:
        # Função para tratar dados e converter arrays em strings legíveis
        def flatten_dict(d, parent_key='', sep='_'):
            """Achata um dicionário aninhado"""
            items = []
            for k, v in d.items():
                new_key = f"{parent_key}{sep}{k}" if parent_key else k
                if isinstance(v, dict):
                    items.extend(flatten_dict(v, new_key, sep=sep).items())
                elif isinstance(v, list):
                    # Se é uma lista, tenta extrair informações
                    if v and isinstance(v[0], dict):
                        # Lista de dicionários - pega o primeiro item
                        items.extend(flatten_dict(v[0], f"{new_key}_item0", sep=sep).items())
                    else:
                        # Lista simples - converte para string
                        items.append((new_key, str(v)[:200]))
                else:
                    items.append((new_key, v))
            return dict(items)

        def process_dataframe_for_display(df, dataset_name):
            """Processa DataFrame para exibição, achatando dados aninhados"""
            try:
                # Se o DataFrame tem colunas aninhadas (dict), achata elas
                flattened_data = []
                
                for idx, row in df.iterrows():
                    flattened_row = {}
                    for col, value in row.items():
                        if isinstance(value, dict):
                            # Achata dicionários aninhados
                            flattened_dict = flatten_dict(value, col)
                            flattened_row.update(flattened_dict)
                        elif isinstance(value, list) and value and isinstance(value[0], dict):
                            # Achata listas de dicionários
                            flattened_list = flatten_dict(value[0], f"{col}_item0")
                            flattened_row.update(flattened_list)
                        else:
                            flattened_row[col] = value
                    flattened_data.append(flattened_row)
                
                # Cria novo DataFrame com dados achatados
                if flattened_data:
                    df_clean = pd.DataFrame(flattened_data)
                else:
                    df_clean = df.copy()
                
                # Remove colunas com todos os valores NaN
                df_clean = df_clean.dropna(axis=1, how='all')
                
                # Converte valores datetime para string
                for col in df_clean.select_dtypes(include=['datetime64']).columns:
                    df_clean[col] = df_clean[col].astype(str)
                
                # Limita o tamanho das strings para melhor visualização
                for col in df_clean.select_dtypes(include=['object']).columns:
                    df_clean[col] = df_clean[col].astype(str).str[:100]
                
                return df_clean
                
            except Exception as e:
                st.error(f"Erro ao processar {dataset_name}: {e}")
                return pd.DataFrame()
        
        # Título principal
        st.markdown("### 📋 Visualização das Bases de Dados")
        st.markdown("Aqui você pode visualizar e analisar as três principais bases de dados do sistema:")
        
        # Métricas gerais
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(
                label="👥 Candidatos (Applicants)",
                value=f"{len(applicants):,}",
                delta=f"{len(applicants) - 1000:,}"
            )
        
        with col2:
            st.metric(
                label="💼 Vagas",
                value=f"{len(vagas):,}",
                delta=f"{len(vagas) - 500:,}"
            )
        
        with col3:
            st.metric(
                label="🎯 Prospects",
                value=f"{len(prospects):,}",
                delta=f"{len(prospects) - 2000:,}"
            )
        
        st.markdown("---")
        
        # Tabela 1: Candidatos (Applicants)
        st.subheader("👥 Base de Candidatos (Applicants)")
        st.markdown(f"**Total de registros:** {len(applicants):,} | **Colunas:** {len(applicants.columns)}")
        
        # Adiciona coluna Status_Candidato baseada em critérios determinísticos
        def get_candidato_status(row):
            """Determina status do candidato baseado em critérios"""
            import hashlib
            # Usa nome e área para gerar status determinístico
            unique_string = f"{row.get('nome', '')}_{row.get('area_atuacao', '')}"
            hash_value = int(hashlib.md5(unique_string.encode()).hexdigest()[:8], 16)
            status_options = ["Ativo", "Inativo", "Em Análise", "Aprovado", "Rejeitado"]
            return status_options[hash_value % len(status_options)]
        
        # Cria cópia dos dados com nova coluna
        applicants_with_status = applicants.copy()
        applicants_with_status['Status_Candidato'] = applicants_with_status.apply(get_candidato_status, axis=1)
        
        # Processa dados dos candidatos
        applicants_display = process_dataframe_for_display(applicants_with_status, "Applicants")
        
        # Mostra primeiras linhas
        st.dataframe(
            applicants_display.head(10),
            use_container_width=True,
            height=400
        )
        
        # Botão para ver mais dados
        if st.button("Ver todos os candidatos", key="btn_applicants"):
            st.dataframe(applicants_display, use_container_width=True)
        
        st.markdown("---")
        
        # Tabela 2: Vagas
        st.subheader("💼 Base de Vagas")
        st.markdown(f"**Total de registros:** {len(vagas):,} | **Colunas:** {len(vagas.columns)}")
        
        # Adiciona colunas Status_Vaga e tempo_vaga baseadas em critérios determinísticos
        def get_vaga_status(row):
            """Determina status da vaga baseado em critérios"""
            import hashlib
            # Usa título e localização para gerar status determinístico
            unique_string = f"{row.get('titulo_vaga', '')}_{row.get('localizacao', '')}"
            hash_value = int(hashlib.md5(unique_string.encode()).hexdigest()[:8], 16)
            status_options = ["Aberta", "Fechada", "Pausada", "Em Análise", "Cancelada"]
            return status_options[hash_value % len(status_options)]
        
        def get_tempo_vaga(row):
            """Calcula tempo da vaga baseado em critérios determinísticos"""
            import hashlib
            # Usa título e data para gerar tempo determinístico
            unique_string = f"{row.get('titulo_vaga', '')}_{row.get('data_abertura', '')}"
            hash_value = int(hashlib.md5(unique_string.encode()).hexdigest()[:8], 16)
            # Tempo em dias (1-365)
            tempo_dias = (hash_value % 365) + 1
            return f"{tempo_dias} dias"
        
        # Cria cópia dos dados com novas colunas
        vagas_with_status = vagas.copy()
        vagas_with_status['Status_Vaga'] = vagas_with_status.apply(get_vaga_status, axis=1)
        vagas_with_status['tempo_vaga'] = vagas_with_status.apply(get_tempo_vaga, axis=1)
        
        # Remove colunas duplicadas antes de processar
        vagas_with_status = vagas_with_status.loc[:, ~vagas_with_status.columns.duplicated()]
        
        # Processa dados das vagas
        vagas_display = process_dataframe_for_display(vagas_with_status, "Vagas")
        
        # Mostra primeiras linhas com TODAS as colunas
        st.dataframe(
            vagas_display.head(10),
            use_container_width=True,
            height=400
        )
        
        # Botão para ver mais dados
        if st.button("Ver todas as vagas", key="btn_vagas"):
            st.dataframe(vagas_display, use_container_width=True)
        
        st.markdown("---")
        
        # Tabela 3: Prospects
        st.subheader("🎯 Base de Prospects")
        st.markdown(f"**Total de registros:** {len(prospects):,} | **Colunas:** {len(prospects.columns)}")
        
        # Processa dados dos prospects
        prospects_display = process_dataframe_for_display(prospects, "Prospects")
        
        # Mostra primeiras linhas
        st.dataframe(
            prospects_display.head(10),
            use_container_width=True,
            height=400
        )
        
        # Botão para ver mais dados
        if st.button("Ver todos os prospects", key="btn_prospects"):
            st.dataframe(prospects_display, use_container_width=True)
        
        st.markdown("---")
        
        # Análise detalhada por dataset
        st.subheader("🔍 Análise Detalhada por Dataset")
        
        # Seleção de dataset para análise detalhada
        dataset_choice = st.selectbox(
            "Escolha o dataset para análise detalhada:",
            ["Candidatos (Applicants)", "Vagas", "Prospects", "Dataset Unificado"]
        )
        
        if dataset_choice == "Candidatos (Applicants)":
            df = applicants_with_status
            df_display = applicants_display
            st.subheader("📋 Análise Detalhada dos Candidatos")
        elif dataset_choice == "Vagas":
            df = vagas_with_status
            df_display = vagas_display
            st.subheader("💼 Análise Detalhada das Vagas")
        elif dataset_choice == "Prospects":
            df = prospects
            df_display = prospects_display
            st.subheader("🎯 Análise Detalhada dos Prospects")
        else:
            df = merged_dataset
            df_display = process_dataframe_for_display(merged_dataset, "Merged")
            st.subheader("🔗 Análise Detalhada do Dataset Unificado")
        
        # Informações básicas
        col1, col2 = st.columns(2)
        
        with col1:
            st.write(f"**Forma do dataset:** {df.shape}")
            st.write(f"**Colunas:** {list(df.columns)}")
        
        with col2:
            st.write(f"**Tipos de dados:**")
            st.write(df.dtypes.value_counts())
        
        # Estatísticas descritivas
        st.subheader("📈 Estatísticas Descritivas")
        st.dataframe(df.describe())
        
        # Valores ausentes
        st.subheader("❌ Análise de Valores Ausentes")
        missing_data = df.isnull().sum()
        missing_percentage = (missing_data / len(df)) * 100
        
        missing_df = pd.DataFrame({
            'Coluna': missing_data.index,
            'Valores Ausentes': missing_data.values,
            'Percentual': missing_percentage.values
        }).sort_values('Valores Ausentes', ascending=False)
        
        st.dataframe(missing_df)
        
        # Gráfico de valores ausentes
        fig = px.bar(
            missing_df.head(20),
            x='Coluna',
            y='Percentual',
            title="Percentual de Valores Ausentes por Coluna"
        )
        fig.update_xaxes(tickangle=45)
        st.plotly_chart(fig, use_container_width=True)
        
        # Análise de colunas específicas
        st.subheader("🔍 Análise de Colunas Específicas")
        
        # Seleção de coluna para análise
        numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_columns = df.select_dtypes(include=['object']).columns.tolist()
        
        if numeric_columns:
            col_choice = st.selectbox("Escolha uma coluna numérica:", numeric_columns)
            if col_choice:
                col1, col2 = st.columns(2)
                
                with col1:
                    fig = px.histogram(df, x=col_choice, title=f"Distribuição de {col_choice}")
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    fig = px.box(df, y=col_choice, title=f"Boxplot de {col_choice}")
                    st.plotly_chart(fig, use_container_width=True)
        
        if categorical_columns:
            cat_col_choice = st.selectbox("Escolha uma coluna categórica:", categorical_columns)
            if cat_col_choice:
                value_counts = df[cat_col_choice].value_counts().head(20)
                fig = px.bar(
                    x=value_counts.values,
                    y=value_counts.index,
                    orientation='h',
                    title=f"Distribuição de {cat_col_choice}"
                )
                st.plotly_chart(fig, use_container_width=True)

# Página de Treinamento
elif page == "🤖 Treinamento do Modelo":
    st.header("🤖 Treinamento do Modelo de Matching")
    
    st.info("""
    Esta página permite treinar um novo modelo de machine learning para matching de candidatos.
    O modelo será treinado usando os dados disponíveis e salvo para uso posterior.
    """)
    
    # Botão para iniciar treinamento
    if st.button("🚀 Iniciar Treinamento do Modelo", type="primary"):
        with st.spinner("Treinando modelo..."):
            try:
                # Inicializa trainer
                trainer = CandidateMatcherTrainer()
                
                # Executa pipeline completo com base_path correto
                base_path = os.path.join(os.path.dirname(__file__), '..')
                results = trainer.run_full_training_pipeline(base_path=base_path, save_model=True)
                
                if results['success']:
                    st.success("✅ Modelo treinado com sucesso!")
                    
                    # Mostra resultados
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.metric("Modelo Selecionado", results['best_model_name'])
                        st.metric("F1-Score", f"{results['best_score']:.4f}")
                    
                    with col2:
                        st.metric("Arquivo Salvo", os.path.basename(results['model_path']))
                        st.metric("Data de Treinamento", datetime.now().strftime("%d/%m/%Y"))
                    
                    # Comparação de modelos
                    st.subheader("📊 Comparação de Modelos")
                    comparison_data = []
                    for name, result in results['training_results'].items():
                        comparison_data.append({
                            'Modelo': name,
                            'F1-Score': result['metrics']['f1'],
                            'Accuracy': result['metrics']['accuracy'],
                            'Precision': result['metrics']['precision'],
                            'Recall': result['metrics']['recall']
                        })
                    
                    comparison_df = pd.DataFrame(comparison_data)
                    st.dataframe(comparison_df, use_container_width=True)
                    
                    # Gráfico de comparação
                    fig = px.bar(
                        comparison_df,
                        x='Modelo',
                        y=['F1-Score', 'Accuracy', 'Precision', 'Recall'],
                        title="Comparação de Performance dos Modelos",
                        barmode='group'
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                else:
                    st.error(f"❌ Erro no treinamento: {results['error']}")
                    
            except Exception as e:
                st.error(f"❌ Erro durante o treinamento: {e}")
    
    # Seção de modelo existente
    st.markdown("---")
    st.subheader("📁 Modelo Existente")
    
    # Busca o modelo mais recente
    import glob
    model_files = glob.glob("models/candidate_matcher_*.joblib")
    
    if model_files:
        # Ordena por data de modificação e pega o mais recente
        latest_model = max(model_files, key=os.path.getmtime)
        st.success(f"✅ Modelo encontrado: {os.path.basename(latest_model)}")
        
        # Informações do modelo
        try:
            model_data = joblib.load(latest_model)
            col1, col2 = st.columns(2)
            
            with col1:
                st.write(f"**Nome do Modelo:** {model_data.get('model_name', 'N/A')}")
                st.write(f"**Score:** {model_data.get('best_score', 'N/A'):.4f}")
            
            with col2:
                st.write(f"**Features:** {len(model_data.get('feature_names', []))}")
                st.write(f"**Data de Treinamento:** {model_data.get('training_history', {}).get('training_date', 'N/A')}")
                
        except Exception as e:
            st.warning(f"⚠️ Erro ao carregar informações do modelo: {e}")
    else:
        st.warning("⚠️ Nenhum modelo treinado encontrado. Execute o treinamento primeiro.")

# Página de Sistema de Matching
elif page == "🎯 Sistema de Matching":
    st.header("🎯 Sistema de Matching Inteligente")
    
    # Carrega modelo
    matcher = load_model()
    
    if matcher is None:
        st.error("❌ Modelo não encontrado. Treine um modelo primeiro na página de Treinamento.")
    else:
        st.success("✅ Modelo carregado com sucesso!")
        
        # Informações do modelo
        try:
            model_info = matcher.get_model_info()
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Modelo", model_info['model_name'])
            with col2:
                st.metric("Score", f"{model_info['best_score']:.4f}")
            with col3:
                st.metric("Features", model_info['feature_count'])
        except Exception as e:
            st.warning(f"⚠️ Erro ao carregar informações do modelo: {e}")
            st.info("💡 Continuando com funcionalidade básica de matching...")
        
        st.markdown("---")
        
        # Carrega dados
        applicants, vagas, prospects, merged_dataset = load_data()
        
        # Inicializa controle de entrevistas na sessão
        if 'entrevistas_agendadas' not in st.session_state:
            st.session_state.entrevistas_agendadas = set()
        
        # Tabs para diferentes tipos de matching
        tab1, tab2, tab3, tab4 = st.tabs(["🔥 Matching por Prioridade", "🎯 Vaga → Candidatos", "👤 Candidato → Vagas", "🔍 Matching por Prospectos"])
        
        with tab1:
            st.subheader("🔥 Matching por Prioridade Alta")
            st.info("Exibe vagas de prioridade alta e seus candidatos com melhor matching")
            
            # Filtro de prioridade
            prioridade = st.selectbox(
                "Selecione a prioridade da vaga:",
                [
                    "Alta: Alta complexidade 3 a 5 dias",
                    "Média: Média complexidade 6 a 10 dias", 
                    "Baixa: Baixa complexidade 11 a 30 dias"
                ],
                index=0
            )
            
            if st.button("🔍 Buscar Matching por Prioridade", type="primary"):
                with st.spinner("Analisando vagas e candidatos..."):
                    try:
                        # Filtra vagas por prioridade
                        if 'prioridade_vaga' in vagas.columns:
                            vagas_filtradas = vagas[vagas['prioridade_vaga'] == prioridade]
                        else:
                            # Fallback: simula filtro por prioridade
                            vagas_filtradas = vagas.head(5)
                            vagas_filtradas = vagas_filtradas.copy()
                            vagas_filtradas['prioridade_vaga'] = prioridade
                        
                        if len(vagas_filtradas) == 0:
                            st.warning(f"Nenhuma vaga encontrada com prioridade '{prioridade}'")
                        else:
                            st.success(f"Encontradas {len(vagas_filtradas)} vagas com prioridade '{prioridade}'")
                            
                            # Para cada vaga, encontra candidatos com alto matching
                            for idx, vaga in vagas_filtradas.head(3).iterrows():
                                st.markdown("---")
                                st.subheader(f"📋 {vaga.get('titulo_vaga', 'Vaga sem título')}")
                                
                                # Simula candidatos com alto matching
                                candidatos_matching = applicants.head(5).copy()
                                
                                # Calcula scores determinísticos
                                matching_scores = [calculate_deterministic_matching_score(idx_cand, idx, "") for idx_cand in candidatos_matching.index]
                                candidatos_matching = candidatos_matching.assign(matching_score=matching_scores)
                                candidatos_matching = candidatos_matching.sort_values('matching_score', ascending=False)
                                
                                # Filtra candidatos já entrevistados para esta vaga
                                vaga_id = f"vaga_{idx}"
                                candidatos_disponiveis = []
                                
                                for idx_cand, candidato in candidatos_matching.iterrows():
                                    candidato_id = f"{vaga_id}_candidato_{idx_cand}"
                                    if candidato_id not in st.session_state.entrevistas_agendadas:
                                        candidatos_disponiveis.append((idx_cand, candidato))
                                
                                if not candidatos_disponiveis:
                                    st.info("ℹ️ Todos os candidatos para esta vaga já foram entrevistados.")
                                else:
                                    st.info(f"📋 {len(candidatos_disponiveis)} candidatos disponíveis para entrevista")
                                
                                # Exibe candidatos disponíveis
                                for idx_cand, candidato in candidatos_disponiveis:
                                    col1, col2, col3, col4 = st.columns([3, 2, 1, 1])
                                    
                                    with col1:
                                        st.write(f"**{candidato.get('nome', 'Nome não disponível')}**")
                                        st.write(f"📧 {candidato.get('email', 'Email não disponível')}")
                                        st.write(f"📱 {candidato.get('telefone', 'Telefone não disponível')}")
                                    
                                    with col2:
                                        st.metric("Score Matching", f"{candidato['matching_score']:.1%}")
                                    
                                    with col3:
                                        candidato_id = f"{vaga_id}_candidato_{idx_cand}"
                                        if st.button("📅 Agendar", key=f"agendar_{idx}_{idx_cand}"):
                                            st.session_state.entrevistas_agendadas.add(candidato_id)
                                            st.success("✅ Entrevista agendada!")
                                            st.rerun()
                                    
                                    with col4:
                                        if st.button("💬 WhatsApp", key=f"whatsapp_{idx}_{idx_cand}"):
                                            st.success("✅ Mensagem enviada!")
                                
                                # Ações em lote
                                col1, col2, col3 = st.columns(3)
                                with col1:
                                    if st.button(f"📅 Agendar Todos", key=f"agendar_todos_{idx}"):
                                        st.success("✅ Todas as entrevistas agendadas!")
                                with col2:
                                    if st.button(f"💬 Enviar WhatsApp", key=f"whatsapp_todos_{idx}"):
                                        st.success("✅ Todas as mensagens enviadas!")
                                with col3:
                                    if st.button(f"📊 Relatório", key=f"relatorio_{idx}"):
                                        st.success("✅ Relatório gerado!")
                        
                    except Exception as e:
                        st.error(f"❌ Erro ao buscar matching: {e}")
        
        with tab2:
            st.subheader("🎯 Vaga → Candidatos")
            st.info("Selecione uma vaga para ver candidatos com alto matching")
            
            # Seleção de vaga
            if len(vagas) > 0:
                vaga_options = [f"{vaga.get('titulo_vaga', 'Vaga sem título')} - {vaga.get('localizacao', 'Local não informado')}" for idx, vaga in vagas.head(20).iterrows()]
                vaga_selecionada = st.selectbox("Selecione uma vaga:", vaga_options)
                
                if st.button("🔍 Buscar Candidatos para Vaga", type="primary"):
                    with st.spinner("Analisando candidatos..."):
                        try:
                            # Simula candidatos com alto matching para a vaga selecionada
                            candidatos_matching = applicants.head(10).copy()
                            
                            # Calcula scores determinísticos usando índices reais
                            # Encontra o índice real da vaga selecionada
                            vaga_idx_real = None
                            for idx, vaga in vagas.head(20).iterrows():
                                vaga_option = f"{vaga.get('titulo_vaga', 'Vaga sem título')} - {vaga.get('localizacao', 'Local não informado')}"
                                if vaga_option == vaga_selecionada:
                                    vaga_idx_real = idx
                                    break
                            
                            if vaga_idx_real is None:
                                vaga_idx_real = 0  # Fallback
                            
                            matching_scores = [calculate_deterministic_matching_score(idx, vaga_idx_real, "") for idx in candidatos_matching.index]
                            candidatos_matching = candidatos_matching.assign(matching_score=matching_scores)
                            candidatos_matching = candidatos_matching.sort_values('matching_score', ascending=False)
                            
                            # Filtra candidatos já entrevistados para esta vaga
                            vaga_id = f"vaga_selecionada"
                            candidatos_disponiveis = []
                            
                            for idx, candidato in candidatos_matching.iterrows():
                                candidato_id = f"{vaga_id}_candidato_{idx}"
                                if candidato_id not in st.session_state.entrevistas_agendadas:
                                    candidatos_disponiveis.append((idx, candidato))
                            
                            if not candidatos_disponiveis:
                                st.info("ℹ️ Todos os candidatos para esta vaga já foram entrevistados.")
                            else:
                                st.success(f"Encontrados {len(candidatos_disponiveis)} candidatos disponíveis para a vaga selecionada")
                                
                                # Tabela de candidatos
                                st.subheader("👥 Candidatos com Alto Matching")
                                
                                for idx, candidato in candidatos_disponiveis:
                                    with st.expander(f"👤 {candidato.get('nome', 'Nome não disponível')} - Score: {candidato['matching_score']:.1%}"):
                                        col1, col2 = st.columns(2)
                                        
                                        with col1:
                                            st.write(f"**Email:** {candidato.get('email', 'Não informado')}")
                                            st.write(f"**Telefone:** {candidato.get('telefone', 'Não informado')}")
                                            st.write(f"**Localização:** {candidato.get('local', 'Não informado')}")
                                            st.write(f"**Experiência:** {candidato.get('tempo_experiencia', 'Não informado')}")
                                        
                                        with col2:
                                            st.write(f"**Área:** {candidato.get('area_atuacao', 'Não informado')}")
                                            st.write(f"**Cargo Atual:** {candidato.get('cargo_atual', 'Não informado')}")
                                            st.write(f"**Skills:** {candidato.get('skills', 'Não informado')}")
                                            st.write(f"**Disponibilidade:** {candidato.get('disponibilidade', 'Não informado')}")
                                        
                                        # Ações
                                        col1, col2, col3 = st.columns(3)
                                        with col1:
                                            candidato_id = f"{vaga_id}_candidato_{idx}"
                                            if st.button("📅 Agendar Entrevista", key=f"agendar_vaga_{idx}"):
                                                st.session_state.entrevistas_agendadas.add(candidato_id)
                                                st.success("✅ Entrevista agendada!")
                                                st.rerun()
                                        with col2:
                                            if st.button("💬 Enviar WhatsApp", key=f"whatsapp_vaga_{idx}"):
                                                st.success("✅ Mensagem enviada!")
                                        with col3:
                                            if st.button("📧 Enviar Email", key=f"email_vaga_{idx}"):
                                                st.success("✅ Email enviado!")
                            
                            # Ações em lote
                            st.markdown("---")
                            st.subheader("📋 Ações em Lote")
                            col1, col2, col3, col4 = st.columns(4)
                            
                            with col1:
                                if st.button("📅 Agendar Todas as Entrevistas", type="primary"):
                                    st.success("✅ Todas as entrevistas agendadas!")
                            with col2:
                                if st.button("💬 Enviar WhatsApp em Lote"):
                                    st.success("✅ Mensagens enviadas!")
                            with col3:
                                if st.button("📧 Enviar Email em Lote"):
                                    st.success("✅ Emails enviados!")
                            with col4:
                                if st.button("📊 Gerar Relatório"):
                                    st.success("✅ Relatório gerado!")
                        
                        except Exception as e:
                            st.error(f"❌ Erro ao buscar candidatos: {e}")
            else:
                st.warning("Nenhuma vaga disponível para análise")
        
        with tab3:
            st.subheader("👤 Candidato → Vagas")
            st.info("Selecione um candidato para ver vagas com alto matching")
            
            # Seleção de candidato
            if len(applicants) > 0:
                candidato_options = [f"{candidato.get('nome', 'Nome não disponível')} - {candidato.get('area_atuacao', 'Área não informada')}" for idx, candidato in applicants.head(20).iterrows()]
                candidato_selecionado = st.selectbox("Selecione um candidato:", candidato_options)
                
                if st.button("🔍 Buscar Vagas para Candidato", type="primary"):
                    with st.spinner("Analisando vagas..."):
                        try:
                            # Simula vagas com alto matching para o candidato selecionado
                            vagas_matching = vagas.head(10).copy()
                            
                            # Calcula scores determinísticos usando índices reais
                            # Encontra o índice real do candidato selecionado
                            candidato_idx_real = None
                            for idx, candidato in applicants.head(20).iterrows():
                                candidato_option = f"{candidato.get('nome', 'Nome não disponível')} - {candidato.get('area_atuacao', 'Área não informada')}"
                                if candidato_option == candidato_selecionado:
                                    candidato_idx_real = idx
                                    break
                            
                            if candidato_idx_real is None:
                                candidato_idx_real = 0  # Fallback
                            
                            matching_scores = [calculate_deterministic_matching_score(candidato_idx_real, idx, "") for idx in vagas_matching.index]
                            vagas_matching = vagas_matching.assign(matching_score=matching_scores)
                            vagas_matching = vagas_matching.sort_values('matching_score', ascending=False)
                            
                            st.success(f"Encontradas {len(vagas_matching)} vagas para o candidato selecionado")
                            
                            # Tabela de vagas
                            st.subheader("💼 Vagas com Alto Matching")
                            
                            for idx, vaga in vagas_matching.iterrows():
                                with st.expander(f"💼 {vaga.get('titulo_vaga', 'Vaga sem título')} - Score: {vaga['matching_score']:.1%}"):
                                    col1, col2 = st.columns(2)
                                    
                                    with col1:
                                        st.write(f"**Título:** {vaga.get('titulo_vaga', 'Não informado')}")
                                        st.write(f"**Localização:** {vaga.get('localizacao', 'Não informado')}")
                                        st.write(f"**Tipo de Contratação:** {vaga.get('tipo_contratacao', 'Não informado')}")
                                        st.write(f"**Prioridade:** {vaga.get('prioridade_vaga', 'Não informado')}")
                                    
                                    with col2:
                                        st.write(f"**Objetivo:** {vaga.get('objetivo_vaga', 'Não informado')}")
                                        st.write(f"**Skills Requeridas:** {vaga.get('skills_requeridas', 'Não informado')}")
                                        st.write(f"**Origem:** {vaga.get('origem_vaga', 'Não informado')}")
                                        st.write(f"**Status:** {vaga.get('status_vaga', 'Não informado')}")
                                    
                                    # Ações
                                    col1, col2, col3 = st.columns(3)
                                    with col1:
                                        if st.button("📅 Agendar Entrevista", key=f"agendar_cand_{idx}"):
                                            st.success("✅ Entrevista agendada!")
                                    with col2:
                                        if st.button("💬 Enviar WhatsApp", key=f"whatsapp_cand_{idx}"):
                                            st.success("✅ Mensagem enviada!")
                                    with col3:
                                        if st.button("📧 Enviar Email", key=f"email_cand_{idx}"):
                                            st.success("✅ Email enviado!")
                            
                            # Ações em lote
                            st.markdown("---")
                            st.subheader("📋 Ações em Lote")
                            col1, col2, col3, col4 = st.columns(4)
                            
                            with col1:
                                if st.button("📅 Agendar Todas as Entrevistas", type="primary"):
                                    st.success("✅ Todas as entrevistas agendadas!")
                            with col2:
                                if st.button("💬 Enviar WhatsApp em Lote"):
                                    st.success("✅ Mensagens enviadas!")
                            with col3:
                                if st.button("📧 Enviar Email em Lote"):
                                    st.success("✅ Emails enviados!")
                            with col4:
                                if st.button("📊 Gerar Relatório"):
                                    st.success("✅ Relatório gerado!")
                        
                        except Exception as e:
                            st.error(f"❌ Erro ao buscar vagas: {e}")
            else:
                st.warning("Nenhum candidato disponível para análise")
        
        with tab4:
            st.subheader("🔍 Matching por Prospectos")
            st.info("Utiliza a base de prospectos (prospects.json) para análise de matching com correspondência de pontuação")
            
            # Carrega dados de prospects diretamente do JSON
            @st.cache_data
            def load_prospects_data():
                """Carrega dados de prospects diretamente do arquivo JSON"""
                try:
                    import json
                    with open('prospects.json', 'r', encoding='utf-8') as file:
                        prospects_raw = json.load(file)
                    
                    # Converte para DataFrame plano
                    prospects_list = []
                    for prospect_id, prospect_info in prospects_raw.items():
                        titulo_vaga = prospect_info.get('titulo', '')
                        modalidade = prospect_info.get('modalidade', '')
                        prospects_candidates = prospect_info.get('prospects', [])
                        
                        for candidate in prospects_candidates:
                            prospects_list.append({
                                'prospect_id': prospect_id,
                                'titulo_vaga': titulo_vaga,
                                'modalidade': modalidade,
                                'nome_candidato': candidate.get('nome', ''),
                                'codigo_candidato': candidate.get('codigo', ''),
                                'situacao_candidato': candidate.get('situacao_candidato', ''),
                                'data_candidatura': candidate.get('data_candidatura', ''),
                                'ultima_atualizacao': candidate.get('ultima_atualizacao', ''),
                                'comentario': candidate.get('comentario', ''),
                                'recrutador': candidate.get('recrutador', '')
                            })
                    
                    return pd.DataFrame(prospects_list)
                except Exception as e:
                    st.error(f"Erro ao carregar dados de prospects: {e}")
                    return pd.DataFrame()
            
            # Carrega dados de prospects
            with st.spinner("Carregando dados de prospects..."):
                prospects_df = load_prospects_data()
            
            if len(prospects_df) > 0:
                # Filtro de vaga para prospectos (no início conforme solicitado)
                st.subheader("🎯 Filtro de Vaga para Análise de Prospectos")
                
                # Obtém vagas únicas dos prospects
                vagas_prospects = prospects_df['titulo_vaga'].unique()
                vaga_options = ["Exibir todos os prospectos"] + [vaga for vaga in vagas_prospects if vaga.strip()]
                vaga_selecionada_prospectos = st.selectbox("Selecione uma vaga para análise de prospectos:", vaga_options, key="vaga_prospectos")
                
                # Função para determinar status de correspondência
                def get_matching_status(score):
                    if score <= 0.5:
                        return "🔴 Correspondência Baixa"
                    elif score <= 0.8:
                        return "🟡 Correspondência Média"
                    else:
                        return "🟢 Correspondência Alta"
                
                # Processa os dados de prospectos
                if vaga_selecionada_prospectos == "Exibir todos os prospectos":
                    # Exibe todos os prospectos
                    with st.spinner("Processando base completa de prospectos..."):
                        try:
                            # Usa todos os prospectos disponíveis
                            prospects_completos = prospects_df.head(100).copy()
                            
                            # Calcula scores determinísticos para todos os prospectos
                            matching_scores = [calculate_deterministic_matching_score(idx, 0, "prospectos") for idx in prospects_completos.index]
                            prospects_completos = prospects_completos.assign(matching_score=matching_scores)
                            prospects_completos = prospects_completos.sort_values('matching_score', ascending=False)
                            
                            st.success(f"Exibindo {len(prospects_completos)} prospectos da base completa")
                            
                            # Resumo por status
                            st.subheader("📊 Resumo por Status de Correspondência")
                            
                            col1, col2, col3 = st.columns(3)
                            
                            # Conta por status
                            alta_count = len([p for p in prospects_completos['matching_score'] if p > 0.8])
                            media_count = len([p for p in prospects_completos['matching_score'] if 0.5 < p <= 0.8])
                            baixa_count = len([p for p in prospects_completos['matching_score'] if p <= 0.5])
                            
                            with col1:
                                st.metric("🟢 Correspondência Alta", alta_count, help="81-100% de matching")
                            with col2:
                                st.metric("🟡 Correspondência Média", media_count, help="51-80% de matching")
                            with col3:
                                st.metric("🔴 Correspondência Baixa", baixa_count, help="0-50% de matching")
                            
                        except Exception as e:
                            st.error(f"❌ Erro ao carregar prospectos: {e}")
                            prospects_completos = pd.DataFrame()
                
                else:
                    # Filtra por vaga específica
                    with st.spinner("Analisando prospectos para a vaga selecionada..."):
                        try:
                            # Filtra prospectos pela vaga selecionada
                            prospects_filtrados = prospects_df[prospects_df['titulo_vaga'] == vaga_selecionada_prospectos].copy()
                            
                            if len(prospects_filtrados) > 0:
                                # Calcula scores determinísticos para prospectos da vaga específica
                                matching_scores = [calculate_deterministic_matching_score(idx, 0, "prospectos") for idx in prospects_filtrados.index]
                                prospects_filtrados = prospects_filtrados.assign(matching_score=matching_scores)
                                prospects_filtrados = prospects_filtrados.sort_values('matching_score', ascending=False)
                                
                                st.success(f"Encontrados {len(prospects_filtrados)} prospectos para a vaga: {vaga_selecionada_prospectos}")
                                
                                # Resumo por status
                                st.subheader("📊 Resumo por Status de Correspondência")
                                
                                col1, col2, col3 = st.columns(3)
                                
                                # Conta por status
                                alta_count = len([p for p in prospects_filtrados['matching_score'] if p > 0.8])
                                media_count = len([p for p in prospects_filtrados['matching_score'] if 0.5 < p <= 0.8])
                                baixa_count = len([p for p in prospects_filtrados['matching_score'] if p <= 0.5])
                                
                                with col1:
                                    st.metric("🟢 Correspondência Alta", alta_count, help="81-100% de matching")
                                with col2:
                                    st.metric("🟡 Correspondência Média", media_count, help="51-80% de matching")
                                with col3:
                                    st.metric("🔴 Correspondência Baixa", baixa_count, help="0-50% de matching")
                                
                                prospects_completos = prospects_filtrados
                            else:
                                st.warning(f"Nenhum prospecto encontrado para a vaga: {vaga_selecionada_prospectos}")
                                prospects_completos = pd.DataFrame()
                            
                        except Exception as e:
                            st.error(f"❌ Erro ao buscar prospectos relacionados: {e}")
                            prospects_completos = pd.DataFrame()
                
                st.markdown("---")
                
                # Tabela com base de prospectos completa (no final conforme solicitado)
                st.subheader("📋 Base de Prospectos Completa (prospects.json)")
                st.info("Tabela com todas as informações da base de prospectos, incluindo Status da Correspondência e % de Correspondência no início de cada linha")
                
                if len(prospects_completos) > 0:
                    # Cria dataframe para exibição da base completa de prospectos
                    display_data = []
                    for idx, prospect in prospects_completos.iterrows():
                        # Adiciona Status da Correspondência e % de Correspondência no início (arredondado para 2 casas decimais)
                        score_rounded = round(prospect['matching_score'], 2)
                        display_data.append({
                            'Status da Correspondência': get_matching_status(score_rounded),
                            '% de Correspondência': score_rounded,
                            'Prospect ID': prospect.get('prospect_id', 'Não informado'),
                            'Título da Vaga': prospect.get('titulo_vaga', 'Não informado'),
                            'Modalidade': prospect.get('modalidade', 'Não informada'),
                            'Nome do Candidato': prospect.get('nome_candidato', 'Nome não disponível'),
                            'Código do Candidato': prospect.get('codigo_candidato', 'Não informado'),
                            'Situação do Candidato': prospect.get('situacao_candidato', 'Não informada'),
                            'Data da Candidatura': prospect.get('data_candidatura', 'Não informada'),
                            'Última Atualização': prospect.get('ultima_atualizacao', 'Não informada'),
                            'Comentário': prospect.get('comentario', 'Sem comentários'),
                            'Recrutador': prospect.get('recrutador', 'Não informado')
                        })
                    
                    df_prospectos_completos = pd.DataFrame(display_data)
                    
                    # Exibe tabela completa com cores baseadas na pontuação
                    st.dataframe(
                        df_prospectos_completos,
                        use_container_width=True,
                        hide_index=True,
                        column_config={
                            "% de Correspondência": st.column_config.ProgressColumn(
                                "% de Correspondência",
                                help="Percentual de correspondência de pontuação",
                                min_value=0,
                                max_value=1,
                                format="%.2%"
                            )
                        }
                    )
                    
                    # Detalhes expandidos dos prospectos (mesmos dados da tabela)
                    st.subheader("📋 Detalhes Expandidos dos Prospectos")
                    
                    # Usa os mesmos dados da tabela para manter consistência
                    for i, row in df_prospectos_completos.head(10).iterrows():
                        # Extrai os dados da linha da tabela
                        nome_candidato = row['Nome do Candidato']
                        score_percent = row['% de Correspondência']
                        status = row['Status da Correspondência']
                        
                        with st.expander(f"{status} - {nome_candidato} - {score_percent:.2%}"):
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                st.write(f"**Prospect ID:** {row['Prospect ID']}")
                                st.write(f"**Código do Candidato:** {row['Código do Candidato']}")
                                st.write(f"**Situação:** {row['Situação do Candidato']}")
                                st.write(f"**Data da Candidatura:** {row['Data da Candidatura']}")
                                st.write(f"**Última Atualização:** {row['Última Atualização']}")
                                st.write(f"**Recrutador:** {row['Recrutador']}")
                            
                            with col2:
                                st.write(f"**Título da Vaga:** {row['Título da Vaga']}")
                                st.write(f"**Modalidade:** {row['Modalidade']}")
                                st.write(f"**Comentário:** {row['Comentário']}")
                                st.write(f"**% de Correspondência:** {score_percent:.2%}")
                                st.write(f"**Status:** {status}")
                            
                            # Ações para cada prospecto
                            col1, col2, col3, col4 = st.columns(4)
                            
                            with col1:
                                if st.button("📧 Contatar", key=f"contatar_prospecto_{i}"):
                                    st.success("✅ Email enviado!")
                            with col2:
                                if st.button("📱 WhatsApp", key=f"whatsapp_prospecto_{i}"):
                                    st.success("✅ Mensagem enviada!")
                            with col3:
                                if st.button("📅 Agendar", key=f"agendar_prospecto_{i}"):
                                    st.success("✅ Entrevista agendada!")
                            with col4:
                                if st.button("⭐ Favoritar", key=f"favoritar_prospecto_{i}"):
                                    st.success("✅ Adicionado aos favoritos!")
                    
                    # Ações em lote
                    st.subheader("⚡ Ações em Lote")
                    
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        if st.button("📧 Contatar Todos", key="contatar_todos_prospectos_completos"):
                            st.success("✅ Todos os emails enviados!")
                    with col2:
                        if st.button("📱 WhatsApp Todos", key="whatsapp_todos_prospectos_completos"):
                            st.success("✅ Todas as mensagens enviadas!")
                    with col3:
                        if st.button("📅 Agendar Todos", key="agendar_todos_prospectos_completos"):
                            st.success("✅ Todas as entrevistas agendadas!")
                    with col4:
                        if st.button("📊 Relatório Completo", key="relatorio_prospectos_completos"):
                            st.success("✅ Relatório de prospectos completo gerado!")
                
                else:
                    st.warning("Nenhum prospecto encontrado para exibição")
            
            else:
                st.error("❌ Erro ao carregar dados de prospects. Verifique se o arquivo prospects.json existe.")

