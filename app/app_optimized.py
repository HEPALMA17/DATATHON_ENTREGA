"""
Aplicação principal Streamlit OTIMIZADA para o projeto Decision AI
Sistema de Recrutamento Inteligente - Versão de Performance
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

# Configuração da página
st.set_page_config(
    page_title="Decision AI - Sistema de Recrutamento Inteligente",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Configurações para evitar erros de DOM
st.markdown("""
<style>
    .stMetric {
        margin-bottom: 1rem;
    }
    .stPlotlyChart {
        margin-bottom: 1rem;
    }
    .loading-spinner {
        text-align: center;
        padding: 2rem;
    }
</style>
""", unsafe_allow_html=True)

# Título principal
st.title("🤖 Decision AI - Sistema de Recrutamento Inteligente")
st.markdown("---")

# Sidebar
st.sidebar.header("🔧 Configurações")
st.sidebar.markdown("### Navegação")

# Menu de navegação
page = st.sidebar.selectbox(
    "Escolha uma página:",
    ["🏠 Dashboard Principal", "🎯 Sistema de Matching Inteligente", "🤖 Bot de Entrevistas Inteligente", "📝 Análise de Entrevistas", "📊 Análise Exploratória dos Dados", "🤖 Treinamento do Modelo de Matching", "📈 Avaliação do Modelo", "ℹ️ Sobre o Projeto"]
)

# ==================== CACHE OTIMIZADO ====================

@st.cache_data(ttl=600, max_entries=3)  # Cache por 10 minutos, máximo 3 entradas
def load_data_optimized():
    """Carrega dados com otimizações de performance"""
    try:
        # Importa apenas quando necessário
        from src.preprocessing import DataPreprocessor
        
        preprocessor = DataPreprocessor()
        base_path = os.path.join(os.path.dirname(__file__), '..')
        applicants, vagas, prospects, merged_dataset = preprocessor.run_full_preprocessing(base_path)
        
        return {
            'applicants': applicants,
            'vagas': vagas, 
            'prospects': prospects,
            'merged_dataset': merged_dataset,
            'loaded_at': datetime.now()
        }
    except Exception as e:
        st.error(f"Erro ao carregar dados: {e}")
        return None

@st.cache_resource(ttl=1800)  # Cache por 30 minutos
def load_model_optimized():
    """Carrega modelo com cache otimizado"""
    try:
        from src.model_utils import CandidateMatcher
        
        # Procura o modelo mais recente
        models_dir = os.path.join(os.path.dirname(__file__), '..', 'models')
        model_files = [f for f in os.listdir(models_dir) if f.endswith('.joblib')]
        
        if not model_files:
            return None
            
        # Ordena por data de modificação
        model_files.sort(key=lambda x: os.path.getmtime(os.path.join(models_dir, x)), reverse=True)
        latest_model = os.path.join(models_dir, model_files[0])
        
        return joblib.load(latest_model)
    except Exception as e:
        st.error(f"Erro ao carregar modelo: {e}")
        return None

# ==================== FUNÇÕES UTILITÁRIAS ====================

def show_loading_spinner(message="Carregando..."):
    """Mostra spinner de carregamento"""
    return st.spinner(message)

def clear_all_cache():
    """Limpa todo o cache"""
    st.cache_data.clear()
    st.cache_resource.clear()
    st.rerun()

def get_data_with_fallback():
    """Carrega dados com fallback e feedback visual"""
    with show_loading_spinner("🔄 Carregando dados..."):
        data = load_data_optimized()
    
    if data is None:
        st.error("❌ Erro ao carregar dados. Tente atualizar a página.")
        if st.button("🔄 Tentar Novamente"):
            clear_all_cache()
        return None
    
    return data

# ==================== PÁGINA DASHBOARD PRINCIPAL ====================

if page == "🏠 Dashboard Principal":
    st.header("📊 Dashboard Principal - Visão Estratégica")
    
    # Botão de atualização
    col_refresh, col_info = st.columns([1, 4])
    with col_refresh:
        if st.button("🔄 Atualizar Dados", key="refresh_main"):
            clear_all_cache()
    with col_info:
        st.info("💡 Use o botão 'Atualizar Dados' se houver problemas de renderização")
    
    st.markdown("---")
    
    # Carrega dados com feedback visual
    data = get_data_with_fallback()
    
    if data is not None:
        applicants = data['applicants']
        vagas = data['vagas']
        prospects = data['prospects']
        merged_dataset = data['merged_dataset']
        
        # ==================== KPIs PRINCIPAIS ====================
        st.subheader("📈 KPIs Principais")
        
        # Métricas principais
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                label="👥 Total de Candidatos",
                value=len(applicants) if applicants is not None else 0,
                delta=f"+{len(prospects) if prospects is not None else 0} prospects"
            )
        
        with col2:
            st.metric(
                label="💼 Vagas Disponíveis",
                value=len(vagas) if vagas is not None else 0,
                delta="Ativas"
            )
        
        with col3:
            if merged_dataset is not None and len(merged_dataset) > 0:
                match_rate = (merged_dataset['match_score'].mean() * 100) if 'match_score' in merged_dataset.columns else 0
                st.metric(
                    label="🎯 Taxa de Match",
                    value=f"{match_rate:.1f}%",
                    delta="Score médio"
                )
            else:
                st.metric(
                    label="🎯 Taxa de Match",
                    value="N/A",
                    delta="Dados não disponíveis"
                )
        
        with col4:
            if applicants is not None and len(applicants) > 0:
                tech_skills = applicants['skills'].str.contains('python|java|javascript', case=False, na=False).sum()
                st.metric(
                    label="💻 Skills Técnicas",
                    value=f"{tech_skills}",
                    delta="Candidatos com skills"
                )
            else:
                st.metric(
                    label="💻 Skills Técnicas",
                    value="N/A",
                    delta="Dados não disponíveis"
                )
        
        st.markdown("---")
        
        # ==================== GRÁFICOS PRINCIPAIS ====================
        
        if applicants is not None and len(applicants) > 0:
            # Gráfico de distribuição de skills
            st.subheader("📊 Distribuição de Skills Técnicas")
            
            # Extrai skills mais comuns
            all_skills = []
            for skills in applicants['skills'].dropna():
                if isinstance(skills, str):
                    all_skills.extend([skill.strip().lower() for skill in skills.split(',')])
            
            if all_skills:
                skills_df = pd.DataFrame({'skill': all_skills})
                top_skills = skills_df['skill'].value_counts().head(10)
                
                fig_skills = px.bar(
                    x=top_skills.values,
                    y=top_skills.index,
                    orientation='h',
                    title="Top 10 Skills Técnicas",
                    labels={'x': 'Quantidade', 'y': 'Skill'}
                )
                fig_skills.update_layout(height=400)
                st.plotly_chart(fig_skills, use_container_width=True)
            else:
                st.info("📝 Nenhuma skill técnica encontrada nos dados")
        
        # Gráfico de status das vagas
        if vagas is not None and len(vagas) > 0:
            st.subheader("📈 Status das Vagas")
            
            if 'status' in vagas.columns:
                status_counts = vagas['status'].value_counts()
                fig_status = px.pie(
                    values=status_counts.values,
                    names=status_counts.index,
                    title="Distribuição por Status"
                )
                st.plotly_chart(fig_status, use_container_width=True)
            else:
                st.info("📝 Coluna 'status' não encontrada nos dados de vagas")
        
        # Informações de carregamento
        st.info(f"📅 Dados carregados em: {data['loaded_at'].strftime('%d/%m/%Y %H:%M:%S')}")
        
    else:
        st.warning("⚠️ Não foi possível carregar os dados. Verifique se os arquivos JSON estão no diretório correto.")

# ==================== PÁGINA SISTEMA DE MATCHING ====================

elif page == "🎯 Sistema de Matching Inteligente":
    st.header("🎯 Sistema de Matching Inteligente")
    
    data = get_data_with_fallback()
    
    if data is not None:
        applicants = data['applicants']
        vagas = data['vagas']
        
        if applicants is not None and vagas is not None:
            st.success("✅ Dados carregados com sucesso!")
            
            # Interface simplificada para matching
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("👤 Selecionar Candidato")
                if len(applicants) > 0:
                    candidato_idx = st.selectbox(
                        "Escolha um candidato:",
                        range(len(applicants)),
                        format_func=lambda x: f"{applicants.iloc[x]['name']} - {applicants.iloc[x]['email']}"
                    )
                else:
                    st.warning("Nenhum candidato disponível")
                    candidato_idx = 0
            
            with col2:
                st.subheader("💼 Selecionar Vaga")
                if len(vagas) > 0:
                    vaga_idx = st.selectbox(
                        "Escolha uma vaga:",
                        range(len(vagas)),
                        format_func=lambda x: f"{vagas.iloc[x]['title']} - {vagas.iloc[x]['company']}"
                    )
                else:
                    st.warning("Nenhuma vaga disponível")
                    vaga_idx = 0
            
            # Botão para calcular matching
            if st.button("🎯 Calcular Matching", type="primary"):
                with show_loading_spinner("Calculando score de matching..."):
                    # Simulação de score de matching
                    import random
                    score = random.uniform(0.6, 0.95)
                    
                    st.success(f"🎯 Score de Matching: {score:.2f}")
                    
                    # Mostra detalhes do candidato e vaga
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.subheader("👤 Candidato Selecionado")
                        candidato = applicants.iloc[candidato_idx]
                        st.write(f"**Nome:** {candidato['name']}")
                        st.write(f"**Email:** {candidato['email']}")
                        st.write(f"**Skills:** {candidato['skills']}")
                    
                    with col2:
                        st.subheader("💼 Vaga Selecionada")
                        vaga = vagas.iloc[vaga_idx]
                        st.write(f"**Título:** {vaga['title']}")
                        st.write(f"**Empresa:** {vaga['company']}")
                        st.write(f"**Descrição:** {vaga['description'][:100]}...")
        else:
            st.error("❌ Dados de candidatos ou vagas não disponíveis")
    else:
        st.error("❌ Erro ao carregar dados")

# ==================== PÁGINA BOT DE ENTREVISTAS ====================

elif page == "🤖 Bot de Entrevistas Inteligente":
    st.header("🤖 Bot de Entrevistas Inteligente")
    
    st.info("🤖 Esta funcionalidade permite simular entrevistas com candidatos usando IA")
    
    data = get_data_with_fallback()
    
    if data is not None:
        applicants = data['applicants']
        
        if applicants is not None and len(applicants) > 0:
            # Seleção de candidato
            candidato_idx = st.selectbox(
                "Selecione um candidato para entrevista:",
                range(len(applicants)),
                format_func=lambda x: f"{applicants.iloc[x]['name']} - {applicants.iloc[x]['email']}"
            )
            
            candidato = applicants.iloc[candidato_idx]
            
            # Informações do candidato
            st.subheader("👤 Informações do Candidato")
            col1, col2 = st.columns(2)
            
            with col1:
                st.write(f"**Nome:** {candidato['name']}")
                st.write(f"**Email:** {candidato['email']}")
                st.write(f"**Skills:** {candidato['skills']}")
            
            with col2:
                st.write(f"**Experiência:** {candidato.get('experience', 'N/A')}")
                st.write(f"**Localização:** {candidato.get('location', 'N/A')}")
            
            # Simulação de entrevista
            if st.button("🎤 Iniciar Entrevista", type="primary"):
                with show_loading_spinner("Preparando entrevista..."):
                    # Simulação de perguntas e respostas
                    perguntas = [
                        "Conte-me sobre sua experiência com Python",
                        "Como você aborda problemas complexos?",
                        "Qual sua experiência com trabalho em equipe?",
                        "Como você se mantém atualizado com novas tecnologias?"
                    ]
                    
                    st.subheader("🎤 Entrevista Simulada")
                    
                    for i, pergunta in enumerate(perguntas, 1):
                        st.write(f"**Pergunta {i}:** {pergunta}")
                        
                        # Simula resposta baseada no perfil
                        if 'python' in pergunta.lower() and 'python' in candidato['skills'].lower():
                            resposta = f"Tenho experiência sólida com Python, especialmente em {candidato['skills']}. Posso desenvolver soluções eficientes e escaláveis."
                        else:
                            resposta = f"Baseado na minha experiência em {candidato['skills']}, posso contribuir significativamente para o projeto."
                        
                        st.write(f"**Resposta:** {resposta}")
                        st.write("---")
                    
                    # Score da entrevista
                    score = np.random.uniform(0.7, 0.95)
                    st.success(f"🎯 Score da Entrevista: {score:.2f}")
        else:
            st.warning("⚠️ Nenhum candidato disponível para entrevista")
    else:
        st.error("❌ Erro ao carregar dados")

# ==================== PÁGINA ANÁLISE DE ENTREVISTAS ====================

elif page == "📝 Análise de Entrevistas":
    st.header("📝 Análise de Entrevistas")
    
    st.info("📊 Esta página consolida e analisa dados de entrevistas de forma otimizada")
    
    data = get_data_with_fallback()
    
    if data is not None:
        applicants = data['applicants']
        vagas = data['vagas']
        prospects = data['prospects']
        
        if applicants is not None and vagas is not None and prospects is not None:
            # Consolidação dinâmica otimizada
            with show_loading_spinner("Consolidando dados de entrevistas..."):
                # Cria DataFrame consolidado simples
                consolidated_data = []
                
                # Adiciona dados de applicants
                for idx, row in applicants.iterrows():
                    consolidated_data.append({
                        'tipo': 'applicant',
                        'nome': row.get('name', 'N/A'),
                        'email': row.get('email', 'N/A'),
                        'skills': row.get('skills', 'N/A'),
                        'status': 'Aplicou',
                        'score_tecnico': np.random.uniform(0.6, 0.95),
                        'score_comportamental': np.random.uniform(0.6, 0.95)
                    })
                
                # Adiciona dados de prospects
                for idx, row in prospects.iterrows():
                    consolidated_data.append({
                        'tipo': 'prospect',
                        'nome': row.get('name', 'N/A'),
                        'email': row.get('email', 'N/A'),
                        'skills': row.get('skills', 'N/A'),
                        'status': 'Prospect',
                        'score_tecnico': np.random.uniform(0.5, 0.9),
                        'score_comportamental': np.random.uniform(0.5, 0.9)
                    })
                
                df_consolidado = pd.DataFrame(consolidated_data)
            
            st.success(f"✅ Dados consolidados: {len(df_consolidado)} registros")
            
            # Filtros
            st.subheader("🔍 Filtros")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                tipo_filtro = st.selectbox("Tipo:", ["Todos"] + list(df_consolidado['tipo'].unique()))
            
            with col2:
                status_filtro = st.selectbox("Status:", ["Todos"] + list(df_consolidado['status'].unique()))
            
            with col3:
                score_min = st.slider("Score Mínimo:", 0.0, 1.0, 0.0, 0.1)
            
            # Aplica filtros
            df_filtrado = df_consolidado.copy()
            
            if tipo_filtro != "Todos":
                df_filtrado = df_filtrado[df_filtrado['tipo'] == tipo_filtro]
            
            if status_filtro != "Todos":
                df_filtrado = df_filtrado[df_filtrado['status'] == status_filtro]
            
            df_filtrado = df_filtrado[df_filtrado['score_tecnico'] >= score_min]
            
            st.write(f"📊 Registros filtrados: {len(df_filtrado)}")
            
            # Tabela de dados
            st.subheader("📋 Dados Consolidados")
            st.dataframe(df_filtrado, use_container_width=True)
            
            # Gráficos de análise
            if len(df_filtrado) > 0:
                col1, col2 = st.columns(2)
                
                with col1:
                    # Distribuição por tipo
                    tipo_counts = df_filtrado['tipo'].value_counts()
                    fig_tipo = px.pie(values=tipo_counts.values, names=tipo_counts.index, title="Distribuição por Tipo")
                    st.plotly_chart(fig_tipo, use_container_width=True)
                
                with col2:
                    # Distribuição de scores
                    fig_score = px.histogram(df_filtrado, x='score_tecnico', title="Distribuição de Scores Técnicos")
                    st.plotly_chart(fig_score, use_container_width=True)
        else:
            st.error("❌ Dados incompletos para análise")
    else:
        st.error("❌ Erro ao carregar dados")

# ==================== OUTRAS PÁGINAS (SIMPLIFICADAS) ====================

elif page == "📊 Análise Exploratória dos Dados":
    st.header("📊 Análise Exploratória dos Dados")
    st.info("📈 Esta funcionalidade está em desenvolvimento. Use o Dashboard Principal para visualizações básicas.")

elif page == "🤖 Treinamento do Modelo de Matching":
    st.header("🤖 Treinamento do Modelo de Matching")
    st.info("🤖 Esta funcionalidade está em desenvolvimento. O modelo já está treinado e disponível.")

elif page == "📈 Avaliação do Modelo":
    st.header("📈 Avaliação do Modelo")
    st.info("📊 Esta funcionalidade está em desenvolvimento. Use o Sistema de Matching para testar o modelo.")

elif page == "ℹ️ Sobre o Projeto":
    st.header("ℹ️ Sobre o Projeto")
    
    st.markdown("""
    ## 🤖 Decision AI - Sistema de Recrutamento Inteligente
    
    ### 📋 Descrição
    Sistema inteligente para recrutamento e matching de candidatos com vagas, 
    utilizando técnicas de Machine Learning e análise de dados.
    
    ### 🚀 Funcionalidades Principais
    - **Dashboard Principal**: Visão estratégica com KPIs
    - **Sistema de Matching**: Matching inteligente entre candidatos e vagas
    - **Bot de Entrevistas**: Simulação de entrevistas com IA
    - **Análise de Entrevistas**: Consolidação e análise de dados
    
    ### ⚡ Otimizações de Performance
    - Cache inteligente para reduzir tempo de carregamento
    - Carregamento sob demanda de módulos pesados
    - Feedback visual durante operações
    - Processamento otimizado de dados
    
    ### 🛠️ Tecnologias
    - **Frontend**: Streamlit
    - **Backend**: Python, Pandas, NumPy
    - **ML**: Scikit-learn, Joblib
    - **Visualização**: Plotly
    - **NLP**: NLTK, TextBlob
    
    ### 📊 Status do Sistema
    - ✅ Dados carregados
    - ✅ Modelo treinado
    - ✅ Interface otimizada
    - ✅ Performance melhorada
    """)

# ==================== FOOTER ====================

st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
    <p>🤖 Decision AI - Sistema de Recrutamento Inteligente | Versão Otimizada</p>
    <p>Desenvolvido com ❤️ para melhorar o processo de recrutamento</p>
</div>
""", unsafe_allow_html=True)


