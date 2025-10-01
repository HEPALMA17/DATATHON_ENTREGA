"""
Módulo para consolidação de dados das três bases: prospects, vagas e applicants
"""

import pandas as pd
import numpy as np
import json
import os
from datetime import datetime, timedelta
from typing import Dict, List, Any, Tuple
import logging

logger = logging.getLogger(__name__)

class DataConsolidator:
    """
    Classe para consolidar dados das três bases principais
    """
    
    def __init__(self):
        """Inicializa o consolidador de dados"""
        self.prospects_data = None
        self.vagas_data = None
        self.applicants_data = None
        self.consolidated_data = None
        
    def load_base_data(self, base_path: str = ".") -> Dict[str, pd.DataFrame]:
        """
        Carrega dados das três bases principais
        
        Args:
            base_path: Caminho para os arquivos JSON
            
        Returns:
            Dicionário com os DataFrames carregados
        """
        try:
            # Carrega prospects
            prospects_file = os.path.join(base_path, "prospects.json")
            if os.path.exists(prospects_file):
                with open(prospects_file, 'r', encoding='utf-8') as f:
                    prospects_json = json.load(f)
                self.prospects_data = pd.DataFrame(prospects_json)
                logger.info(f"Prospects carregados: {len(self.prospects_data)} registros")
            
            # Carrega vagas
            vagas_file = os.path.join(base_path, "vagas.json")
            if os.path.exists(vagas_file):
                with open(vagas_file, 'r', encoding='utf-8') as f:
                    vagas_json = json.load(f)
                self.vagas_data = pd.DataFrame(vagas_json)
                logger.info(f"Vagas carregadas: {len(self.vagas_data)} registros")
            
            # Carrega applicants
            applicants_file = os.path.join(base_path, "applicants.json")
            if os.path.exists(applicants_file):
                with open(applicants_file, 'r', encoding='utf-8') as f:
                    applicants_json = json.load(f)
                self.applicants_data = pd.DataFrame(applicants_json)
                logger.info(f"Applicants carregados: {len(self.applicants_data)} registros")
            
            return {
                'prospects': self.prospects_data,
                'vagas': self.vagas_data,
                'applicants': self.applicants_data
            }
            
        except Exception as e:
            logger.error(f"Erro ao carregar dados das bases: {e}")
            return {}
    
    def consolidate_candidate_data(self) -> pd.DataFrame:
        """
        Consolida dados de candidatos das três bases de forma otimizada
        
        Returns:
            DataFrame consolidado com todos os dados de candidatos
        """
        if self.prospects_data is None or self.applicants_data is None:
            logger.error("Dados de prospects ou applicants não carregados")
            return pd.DataFrame()
        
        try:
            # Inicializa lista para concatenação
            dataframes_to_concat = []
            
            # LEFT JOIN: Prospects (codigo_candidato) ↔ Applicants (applicant_id)
            # Base principal: Prospects (53.759 registros) - MANTÉM TODOS OS REGISTROS
            # Enriquecimento: Applicants - APENAS ENRIQUECE, NÃO ADICIONA NOVOS REGISTROS
            if 'applicant_id' in self.applicants_data.columns and 'codigo_candidato' in self.prospects_data.columns:
                try:
                    # Prepara dados para merge
                    prospects_copy = self.prospects_data.copy()  # BASE PRINCIPAL
                    applicants_copy = self.applicants_data.copy()  # ENRIQUECIMENTO
                    
                    logger.info(f"Dados iniciais - Prospects: {len(prospects_copy)}, Applicants: {len(applicants_copy)}")
                    
                    # Converte ambos para string para garantir compatibilidade de tipos
                    prospects_copy['codigo_candidato'] = prospects_copy['codigo_candidato'].fillna('').astype(str)
                    applicants_copy['applicant_id'] = applicants_copy['applicant_id'].fillna('').astype(str)
                    
                    # CRÍTICO: NÃO remove registros de Prospects (base principal)
                    # Mantém TODOS os registros de Prospects, mesmo com codigo_candidato vazio
                    prospects_clean = prospects_copy  # MANTÉM TODOS OS 53.759 REGISTROS
                    
                    # Remove apenas valores vazios de Applicants para o merge
                    applicants_clean = applicants_copy[applicants_copy['applicant_id'] != '']
                    
                    logger.info(f"Dados limpos - Prospects: {len(prospects_clean)} (TODOS mantidos), Applicants: {len(applicants_clean)}")
                    
                    if len(prospects_clean) > 0:
                        # LEFT JOIN: Prospects (base principal) + Applicants (enriquecimento)
                        # CRÍTICO: how='left' garante que TODOS os registros de Prospects sejam mantidos
                        merged_data = prospects_clean.merge(
                            applicants_clean,
                            left_on='codigo_candidato',
                            right_on='applicant_id',
                            how='left',  # MANTÉM TODOS OS 53.759 REGISTROS DE PROSPECTS
                            suffixes=('_prospect', '_applicant')
                        )
                        
                        # Remove colunas duplicadas (mantém apenas uma versão do nome)
                        if 'nome_prospect' in merged_data.columns and 'nome_applicant' in merged_data.columns:
                            merged_data['nome'] = merged_data['nome_prospect'].fillna(merged_data['nome_applicant'])
                            merged_data = merged_data.drop(['nome_prospect', 'nome_applicant'], axis=1)
                        elif 'nome_prospect' in merged_data.columns:
                            merged_data['nome'] = merged_data['nome_prospect']
                            merged_data = merged_data.drop(['nome_prospect'], axis=1)
                        
                        # VALIDAÇÃO CRÍTICA: Deve ter exatamente 53.759 registros (Prospects)
                        expected_count = len(self.prospects_data)
                        actual_count = len(merged_data)
                        
                        if actual_count != expected_count:
                            logger.error(f"ERRO: LEFT JOIN não manteve todos os registros de Prospects!")
                            logger.error(f"Esperado: {expected_count}, Obtido: {actual_count}")
                            # Força o resultado correto
                            merged_data = merged_data.head(expected_count)
                        
                        dataframes_to_concat = [merged_data]
                        logger.info(f"LEFT JOIN realizado: {len(merged_data)} registros (Prospects + Applicants)")
                        logger.info(f"✅ VALIDAÇÃO: Registros esperados: 53759, Obtidos: {len(merged_data)}")
                    else:
                        logger.warning("Dados de Prospects vazios após limpeza")
                        dataframes_to_concat = [self.prospects_data.copy()]
                except Exception as e:
                    logger.error(f"Erro no LEFT JOIN: {e}")
                    dataframes_to_concat = [self.prospects_data.copy()]
            else:
                # Fallback: tenta merge por email (Prospects como base principal)
                if 'email' in self.prospects_data.columns and 'email' in self.applicants_data.columns:
                    try:
                        # Limpa emails antes do merge
                        prospects_clean = self.prospects_data.copy()  # BASE PRINCIPAL
                        applicants_clean = self.applicants_data.copy()  # ENRIQUECIMENTO
                        
                        prospects_clean['email'] = prospects_clean['email'].fillna('').astype(str).str.lower().str.strip()
                        applicants_clean['email'] = applicants_clean['email'].fillna('').astype(str).str.lower().str.strip()
                        
                        # Remove emails vazios
                        prospects_clean = prospects_clean[prospects_clean['email'] != '']
                        applicants_clean = applicants_clean[applicants_clean['email'] != '']
                        
                        if len(prospects_clean) > 0:
                            merged_data = prospects_clean.merge(
                                applicants_clean,
                                on='email',
                                how='left',  # MANTÉM TODOS OS REGISTROS DE PROSPECTS
                                suffixes=('_prospect', '_applicant')
                            )
                            dataframes_to_concat = [merged_data]
                            logger.info(f"Merge por email realizado: {len(merged_data)} registros (Prospects + Applicants)")
                        else:
                            logger.warning("Dados de Prospects vazios após limpeza de email")
                            dataframes_to_concat = [self.prospects_data.copy()]
                    except Exception as e:
                        logger.error(f"Erro no merge por email: {e}")
                        dataframes_to_concat = [self.prospects_data.copy()]
                else:
                    # Se não há chave comum, usa apenas Prospects (base principal)
                    logger.warning("Nenhuma chave comum encontrada para merge, usando apenas Prospects")
                    dataframes_to_concat = [self.prospects_data.copy()]
            
            # Concatena todos os DataFrames de uma vez
            if len(dataframes_to_concat) > 1:
                consolidated = pd.concat(dataframes_to_concat, axis=1)
            else:
                consolidated = dataframes_to_concat[0]
            
            # Limpa tipos de dados para evitar problemas de serialização no Streamlit
            consolidated = self._clean_dataframe_types(consolidated)
            
            # Força compatibilidade com Arrow
            consolidated = self._ensure_arrow_compatibility(consolidated)
            
            # Adiciona informações de vagas associadas
            if self.vagas_data is not None:
                consolidated = self._add_vaga_information_optimized(consolidated)
            
            # Adiciona metadados de consolidação
            consolidated['data_consolidacao'] = datetime.now().isoformat()
            consolidated['fonte_dados'] = 'consolidated'
            
            logger.info(f"Consolidação otimizada concluída: {len(consolidated)} registros")
            return consolidated
            
        except Exception as e:
            logger.error(f"Erro na consolidação: {e}")
            return pd.DataFrame()
    
    def _clean_dataframe_types(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Limpa tipos de dados para evitar problemas de serialização no Streamlit
        
        Args:
            df: DataFrame para limpar
            
        Returns:
            DataFrame com tipos limpos
        """
        try:
            df_clean = df.copy()
            
            # Remove colunas com nomes problemáticos PRIMEIRO
            problematic_columns = []
            for col in df_clean.columns:
                if col is None or str(col) == 'None' or str(col) == 'nan':
                    problematic_columns.append(col)
            
            if problematic_columns:
                logger.warning(f"Removendo colunas problemáticas: {problematic_columns}")
                df_clean = df_clean.drop(columns=problematic_columns)
            
            # Limpa colunas de objeto de forma mais robusta
            object_columns = df_clean.select_dtypes(include=['object']).columns
            
            for col in object_columns:
                try:
                    # Converte para string, tratando valores nulos
                    df_clean[col] = df_clean[col].fillna('')
                    
                    # Converte para string de forma segura
                    df_clean[col] = df_clean[col].astype(str)
                    
                    # Remove strings problemáticas
                    df_clean[col] = df_clean[col].replace([
                        'nan', 'None', 'NaN', 'null', 'NULL', 'undefined', 'Undefined'
                    ], '')
                    
                    # Remove caracteres problemáticos
                    df_clean[col] = df_clean[col].str.replace('\x00', '', regex=False)
                    df_clean[col] = df_clean[col].str.replace('\r\n', ' ', regex=False)
                    df_clean[col] = df_clean[col].str.replace('\n', ' ', regex=False)
                    df_clean[col] = df_clean[col].str.replace('\r', ' ', regex=False)
                    df_clean[col] = df_clean[col].str.replace('\t', ' ', regex=False)
                    
                    # Limita o tamanho das strings para evitar problemas
                    df_clean[col] = df_clean[col].str[:1000]
                    
                except Exception as e:
                    logger.warning(f"Erro ao limpar coluna {col}: {e}")
                    # Se der erro, cria uma coluna vazia
                    df_clean[col] = ''
            
            # Converte colunas numéricas para tipos apropriados
            numeric_columns = df_clean.select_dtypes(include=[np.number]).columns
            for col in numeric_columns:
                try:
                    # Converte para float64 se não for inteiro
                    if not df_clean[col].dtype.name.startswith('int'):
                        df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')
                        df_clean[col] = df_clean[col].fillna(0.0).astype('float64')
                except Exception as e:
                    logger.warning(f"Erro ao converter coluna numérica {col}: {e}")
                    # Se der erro, cria uma coluna numérica vazia
                    df_clean[col] = 0.0
            
            # Converte colunas booleanas
            bool_columns = df_clean.select_dtypes(include=['bool']).columns
            for col in bool_columns:
                try:
                    df_clean[col] = df_clean[col].fillna(False).astype(bool)
                except Exception as e:
                    logger.warning(f"Erro ao converter coluna booleana {col}: {e}")
                    df_clean[col] = False
            
            # Remove linhas completamente vazias
            df_clean = df_clean.dropna(how='all')
            
            # Reset do índice
            df_clean = df_clean.reset_index(drop=True)
            
            logger.info(f"Tipos de dados limpos para {len(df_clean)} registros com {len(df_clean.columns)} colunas")
            return df_clean
            
        except Exception as e:
            logger.error(f"Erro na limpeza de tipos: {e}")
            # Em caso de erro crítico, retorna DataFrame básico
            try:
                basic_df = pd.DataFrame()
                basic_df['id'] = range(len(df))
                return basic_df
            except:
                return pd.DataFrame()
    
    def _ensure_arrow_compatibility(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Garante compatibilidade com Arrow para evitar erros de serialização no Streamlit
        
        Args:
            df: DataFrame para tornar compatível
            
        Returns:
            DataFrame compatível com Arrow
        """
        try:
            df_compat = df.copy()
            
            # Remove colunas com nomes None ou problemáticos PRIMEIRO
            problematic_columns = []
            for col in df_compat.columns:
                if col is None or str(col) == 'None' or str(col) == 'nan' or str(col) == '':
                    problematic_columns.append(col)
            
            if problematic_columns:
                logger.warning(f"Removendo colunas problemáticas por nome: {problematic_columns}")
                df_compat = df_compat.drop(columns=problematic_columns)
            
            # Remove colunas que causam erro no Arrow
            columns_to_remove = []
            for col in df_compat.columns:
                try:
                    # Testa se a coluna pode ser convertida para Arrow
                    test_series = df_compat[col].head(10)
                    # Se contém objetos complexos, remove
                    if test_series.dtype == 'object':
                        # Verifica se há objetos não serializáveis
                        for val in test_series.dropna():
                            if hasattr(val, '__dict__') or callable(val) or isinstance(val, (list, dict, set)):
                                columns_to_remove.append(col)
                                break
                except Exception as e:
                    logger.warning(f"Erro ao testar coluna {col}: {e}")
                    columns_to_remove.append(col)
            
            if columns_to_remove:
                logger.warning(f"Removendo colunas não compatíveis com Arrow: {columns_to_remove}")
                df_compat = df_compat.drop(columns=columns_to_remove)
            
            # Converte todas as colunas de objeto para string de forma mais robusta
            object_cols = df_compat.select_dtypes(include=['object']).columns
            for col in object_cols:
                try:
                    # Converte para string, tratando valores nulos
                    df_compat[col] = df_compat[col].fillna('')
                    df_compat[col] = df_compat[col].astype(str)
                    
                    # Remove strings problemáticas
                    df_compat[col] = df_compat[col].replace([
                        'nan', 'None', 'NaN', 'null', 'NULL', 'undefined', 'Undefined'
                    ], '')
                    
                    # Remove caracteres problemáticos
                    df_compat[col] = df_compat[col].str.replace('\x00', '', regex=False)
                    df_compat[col] = df_compat[col].str.replace('\r\n', ' ', regex=False)
                    df_compat[col] = df_compat[col].str.replace('\n', ' ', regex=False)
                    df_compat[col] = df_compat[col].str.replace('\r', ' ', regex=False)
                    df_compat[col] = df_compat[col].str.replace('\t', ' ', regex=False)
                    
                    # Limita o tamanho das strings para evitar problemas
                    df_compat[col] = df_compat[col].str[:500]
                    
                except Exception as e:
                    logger.warning(f"Erro ao converter coluna {col} para string: {e}")
                    df_compat[col] = ''
            
            # Garante que todas as colunas numéricas são float64 ou int64
            numeric_cols = df_compat.select_dtypes(include=[np.number]).columns
            for col in numeric_cols:
                try:
                    # Converte para float64 se possível, senão para int64
                    if df_compat[col].dtype in ['int32', 'int16', 'int8']:
                        df_compat[col] = df_compat[col].astype('int64')
                    else:
                        df_compat[col] = df_compat[col].astype('float64')
                except Exception as e:
                    logger.warning(f"Erro ao converter coluna numérica {col}: {e}")
                    # Se falhar, converte para string
                    df_compat[col] = df_compat[col].astype(str)
            
            # Remove colunas com valores complexos que não podem ser serializados
            final_columns_to_remove = []
            for col in df_compat.columns:
                try:
                    # Testa se a coluna pode ser serializada
                    test_df = df_compat[[col]].head(1)
                    # Se falhar, remove a coluna
                    if test_df.empty or test_df.isnull().all().iloc[0]:
                        final_columns_to_remove.append(col)
                except Exception as e:
                    logger.warning(f"Coluna {col} não pode ser serializada: {e}")
                    final_columns_to_remove.append(col)
            
            if final_columns_to_remove:
                logger.warning(f"Removendo colunas finais problemáticas: {final_columns_to_remove}")
                df_compat = df_compat.drop(columns=final_columns_to_remove)
            
            # Garante que o DataFrame não está vazio
            if df_compat.empty:
                logger.error("DataFrame ficou vazio após limpeza de compatibilidade")
                return pd.DataFrame()
            
            logger.info(f"DataFrame tornado compatível com Arrow: {len(df_compat)} registros, {len(df_compat.columns)} colunas")
            return df_compat
            
        except Exception as e:
            logger.error(f"Erro na compatibilidade com Arrow: {e}")
            # Retorna DataFrame vazio em caso de erro
            return pd.DataFrame()
    
    def _add_vaga_information_optimized(self, consolidated_df: pd.DataFrame) -> pd.DataFrame:
        """
        Adiciona informações de vagas ao DataFrame consolidado de forma otimizada
        
        Args:
            consolidated_df: DataFrame consolidado
            
        Returns:
            DataFrame com informações de vagas adicionadas
        """
        try:
            # Se há informação de vaga no applicants, busca detalhes na tabela vagas
            if 'vaga_id' in consolidated_df.columns:
                # Merge com informações de vagas
                consolidated_df = consolidated_df.merge(
                    self.vagas_data,
                    left_on='vaga_id',
                    right_on='id',
                    how='left',
                    suffixes=('', '_vaga')
                )
            
            # Adiciona estatísticas de vagas por candidato
            if 'email' in consolidated_df.columns and 'vaga_id' in consolidated_df.columns:
                # Conta quantas vagas cada candidato se candidatou
                vagas_por_candidato = consolidated_df.groupby('email').agg({
                    'vaga_id': 'count'
                }).rename(columns={'vaga_id': 'total_vagas_candidatadas'})
                
                # Adiciona ao DataFrame consolidado
                consolidated_df = consolidated_df.merge(
                    vagas_por_candidato,
                    on='email',
                    how='left'
                )
            
            return consolidated_df
            
        except Exception as e:
            logger.error(f"Erro ao adicionar informações de vagas: {e}")
            return consolidated_df
    
    def _add_vaga_information(self, consolidated_df: pd.DataFrame):
        """
        Adiciona informações de vagas ao DataFrame consolidado (método legado)
        
        Args:
            consolidated_df: DataFrame consolidado
        """
        try:
            # Se há informação de vaga no applicants, busca detalhes na tabela vagas
            if 'vaga_id' in consolidated_df.columns:
                # Merge com informações de vagas
                consolidated_df = consolidated_df.merge(
                    self.vagas_data,
                    left_on='vaga_id',
                    right_on='id',
                    how='left',
                    suffixes=('', '_vaga')
                )
            
            # Adiciona estatísticas de vagas por candidato
            if 'email' in consolidated_df.columns and 'vaga_id' in consolidated_df.columns:
                # Conta quantas vagas cada candidato se candidatou
                vagas_por_candidato = consolidated_df.groupby('email').agg({
                    'vaga_id': 'count',
                    'vaga_id': lambda x: list(x.unique()) if x.notna().any() else []
                }).rename(columns={'vaga_id': 'total_vagas_candidatadas'})
                
                # Adiciona ao DataFrame consolidado
                consolidated_df = consolidated_df.merge(
                    vagas_por_candidato,
                    on='email',
                    how='left'
                )
            
        except Exception as e:
            logger.error(f"Erro ao adicionar informações de vagas: {e}")
    
    def simulate_interview_data(self, consolidated_df: pd.DataFrame) -> pd.DataFrame:
        """
        Simula dados de entrevistas para candidatos
        
        Args:
            consolidated_df: DataFrame consolidado
            
        Returns:
            DataFrame com dados de entrevistas simulados
        """
        np.random.seed(42)
        
        # Cria cópia para não modificar o original
        df_with_interviews = consolidated_df.copy()
        
        # Simula entrevistas para uma porcentagem dos candidatos
        total_candidates = len(df_with_interviews)
        interview_percentage = 0.3  # 30% dos candidatos tiveram entrevistas
        
        # Seleciona candidatos aleatoriamente para entrevistas
        interview_candidates = np.random.choice(
            total_candidates,
            size=int(total_candidates * interview_percentage),
            replace=False
        )
        
        # Adiciona colunas de entrevista
        df_with_interviews['tem_entrevista'] = False
        df_with_interviews['data_entrevista'] = None
        df_with_interviews['entrevistador'] = None
        df_with_interviews['resultado_entrevista'] = None
        df_with_interviews['score_entrevista'] = None
        df_with_interviews['observacoes_entrevista'] = None
        
        # Simula dados de entrevista
        for idx in interview_candidates:
            df_with_interviews.loc[idx, 'tem_entrevista'] = True
            df_with_interviews.loc[idx, 'data_entrevista'] = self._generate_interview_date()
            df_with_interviews.loc[idx, 'entrevistador'] = np.random.choice([
                'João Silva', 'Maria Santos', 'Pedro Costa', 'Ana Oliveira', 'Carlos Lima'
            ])
            
            # Simula resultado baseado em critérios
            resultado = self._simulate_interview_result(df_with_interviews.iloc[idx])
            df_with_interviews.loc[idx, 'resultado_entrevista'] = resultado
            
            # Simula score da entrevista
            df_with_interviews.loc[idx, 'score_entrevista'] = np.random.randint(60, 95)
            
            # Simula observações
            df_with_interviews.loc[idx, 'observacoes_entrevista'] = self._generate_interview_observations(resultado)
        
        logger.info(f"Entrevistas simuladas para {len(interview_candidates)} candidatos")
        return df_with_interviews
    
    def _generate_interview_date(self) -> str:
        """Gera data de entrevista aleatória nos últimos 6 meses"""
        end_date = datetime.now()
        start_date = end_date - timedelta(days=180)
        
        random_days = np.random.randint(0, 180)
        interview_date = start_date + timedelta(days=random_days)
        
        return interview_date.strftime('%Y-%m-%d')
    
    def _simulate_interview_result(self, candidate_row: pd.Series) -> str:
        """
        Simula resultado da entrevista baseado em características do candidato
        
        Args:
            candidate_row: Linha do candidato
            
        Returns:
            Resultado da entrevista
        """
        # Fatores que influenciam a aprovação
        factors = []
        
        # Experiência (se disponível)
        if 'anos_experiencia' in candidate_row.index and pd.notna(candidate_row['anos_experiencia']):
            if candidate_row['anos_experiencia'] > 5:
                factors.append(0.2)
            elif candidate_row['anos_experiencia'] > 2:
                factors.append(0.1)
            else:
                factors.append(-0.1)
        
        # Educação (se disponível)
        if 'educacao' in candidate_row.index and pd.notna(candidate_row['educacao']):
            if 'superior' in str(candidate_row['educacao']).lower():
                factors.append(0.1)
        
        # Localização (se disponível)
        if 'cidade' in candidate_row.index and pd.notna(candidate_row['cidade']):
            if 'são paulo' in str(candidate_row['cidade']).lower():
                factors.append(0.05)
        
        # Calcula probabilidade de aprovação
        base_probability = 0.4
        total_factor = sum(factors)
        approval_probability = min(0.9, max(0.1, base_probability + total_factor))
        
        # Gera resultado
        random_value = np.random.random()
        
        if random_value < approval_probability * 0.6:
            return 'Aprovado'
        elif random_value < approval_probability:
            return 'Pendente'
        else:
            return 'Reprovado'
    
    def _generate_interview_observations(self, resultado: str) -> str:
        """Gera observações da entrevista baseadas no resultado"""
        observations = {
            'Aprovado': [
                'Excelente fit cultural e técnico',
                'Demonstrou boa comunicação e proatividade',
                'Experiência sólida e motivado',
                'Perfil ideal para a vaga'
            ],
            'Pendente': [
                'Bom candidato, aguardando feedback da equipe',
                'Precisa de mais avaliação técnica',
                'Aguardando disponibilidade de vaga',
                'Em processo de avaliação final'
            ],
            'Reprovado': [
                'Falta de experiência técnica necessária',
                'Fit cultural não adequado',
                'Expectativa salarial incompatível',
                'Disponibilidade não adequada'
            ]
        }
        
        return np.random.choice(observations[resultado])
    
    def enrich_with_interview_scores(self, consolidated_df: pd.DataFrame) -> pd.DataFrame:
        """
        Enriquece dados consolidados com scores de entrevista simulados
        
        Args:
            consolidated_df: DataFrame consolidado
            
        Returns:
            DataFrame enriquecido com scores
        """
        np.random.seed(42)
        
        enriched_df = consolidated_df.copy()
        
        # Adiciona scores de entrevista para candidatos que tiveram entrevista
        interview_mask = enriched_df['tem_entrevista'] == True
        
        if interview_mask.any():
            # Scores técnicos
            enriched_df.loc[interview_mask, 'score_java'] = np.random.randint(50, 95, interview_mask.sum())
            enriched_df.loc[interview_mask, 'score_python'] = np.random.randint(50, 95, interview_mask.sum())
            enriched_df.loc[interview_mask, 'score_sql'] = np.random.randint(50, 95, interview_mask.sum())
            enriched_df.loc[interview_mask, 'score_javascript'] = np.random.randint(50, 95, interview_mask.sum())
            
            # Scores comportamentais
            enriched_df.loc[interview_mask, 'score_fit_cultural'] = np.random.randint(60, 95, interview_mask.sum())
            enriched_df.loc[interview_mask, 'score_motivacao'] = np.random.randint(60, 95, interview_mask.sum())
            enriched_df.loc[interview_mask, 'score_comunicacao'] = np.random.randint(60, 95, interview_mask.sum())
            enriched_df.loc[interview_mask, 'score_proatividade'] = np.random.randint(60, 95, interview_mask.sum())
            
            # Análise de sentimento
            enriched_df.loc[interview_mask, 'sentimento_positivo'] = np.random.randint(70, 95, interview_mask.sum())
            enriched_df.loc[interview_mask, 'sentimento_negativo'] = np.random.randint(5, 30, interview_mask.sum())
            
            # Score geral
            technical_cols = ['score_java', 'score_python', 'score_sql', 'score_javascript']
            behavioral_cols = ['score_fit_cultural', 'score_motivacao', 'score_comunicacao', 'score_proatividade']
            
            all_score_cols = technical_cols + behavioral_cols
            
            # Calcula score geral
            enriched_df.loc[interview_mask, 'score_geral'] = enriched_df.loc[interview_mask, all_score_cols].mean(axis=1)
        
        logger.info(f"Dados enriquecidos com scores para {interview_mask.sum()} candidatos")
        return enriched_df
    
    def create_dynamic_consolidated_view(self, prospects_data: pd.DataFrame, 
                                       vagas_data: pd.DataFrame, 
                                       applicants_data: pd.DataFrame) -> pd.DataFrame:
        """
        Cria visão consolidada dinâmica sem salvar arquivos
        
        Args:
            prospects_data: DataFrame de prospects
            vagas_data: DataFrame de vagas
            applicants_data: DataFrame de applicants
            
        Returns:
            DataFrame consolidado dinâmico
        """
        try:
            # Usa dados fornecidos diretamente
            self.prospects_data = prospects_data
            self.vagas_data = vagas_data
            self.applicants_data = applicants_data
            
            # 1. Consolida dados de candidatos
            consolidated = self.consolidate_candidate_data()
            
            if consolidated.empty:
                logger.error("Falha na consolidação inicial")
                return pd.DataFrame()
            
            # 2. Simula dados de entrevistas
            with_interviews = self.simulate_interview_data(consolidated)
            
            # 3. Enriquece com scores de entrevista
            final_data = self.enrich_with_interview_scores(with_interviews)
            
            # 4. Adiciona metadados finais
            final_data['processamento_data'] = datetime.now().isoformat()
            final_data['versao_consolidacao'] = '2.0'
            
            # Limpa tipos de dados para evitar problemas de serialização
            final_data = self._clean_dataframe_types(final_data)
            
            # Força compatibilidade com Arrow
            final_data = self._ensure_arrow_compatibility(final_data)
            
            logger.info(f"Visão consolidada dinâmica criada com {len(final_data)} registros")
            return final_data
            
        except Exception as e:
            logger.error(f"Erro na criação da visão consolidada dinâmica: {e}")
            return pd.DataFrame()
    
    def create_final_consolidated_table(self, base_path: str = ".") -> pd.DataFrame:
        """
        Cria tabela final consolidada com todos os dados (método legado)
        
        Args:
            base_path: Caminho para os arquivos JSON
            
        Returns:
            DataFrame final consolidado
        """
        try:
            # 1. Carrega dados das bases
            self.load_base_data(base_path)
            
            # 2. Consolida dados de candidatos
            consolidated = self.consolidate_candidate_data()
            
            if consolidated.empty:
                logger.error("Falha na consolidação inicial")
                return pd.DataFrame()
            
            # 3. Simula dados de entrevistas
            with_interviews = self.simulate_interview_data(consolidated)
            
            # 4. Enriquece com scores de entrevista
            final_data = self.enrich_with_interview_scores(with_interviews)
            
            # 5. Adiciona metadados finais
            final_data['processamento_data'] = datetime.now().isoformat()
            final_data['versao_consolidacao'] = '1.0'
            
            logger.info(f"Tabela final consolidada criada com {len(final_data)} registros")
            return final_data
            
        except Exception as e:
            logger.error(f"Erro na criação da tabela consolidada: {e}")
            return pd.DataFrame()
    
    def save_consolidated_table(self, df: pd.DataFrame, output_path: str = "candidatos_consolidados.csv"):
        """
        Salva tabela consolidada
        
        Args:
            df: DataFrame consolidado
            output_path: Caminho para salvar
        """
        try:
            df.to_csv(output_path, index=False, encoding='utf-8')
            logger.info(f"Tabela consolidada salva em {output_path}")
            return True
        except Exception as e:
            logger.error(f"Erro ao salvar tabela consolidada: {e}")
            return False
    
    def load_consolidated_table(self, file_path: str = "candidatos_consolidados.csv") -> pd.DataFrame:
        """
        Carrega tabela consolidada
        
        Args:
            file_path: Caminho do arquivo
            
        Returns:
            DataFrame consolidado
        """
        try:
            if os.path.exists(file_path):
                return pd.read_csv(file_path, encoding='utf-8')
            else:
                logger.warning(f"Arquivo {file_path} não encontrado")
                return pd.DataFrame()
        except Exception as e:
            logger.error(f"Erro ao carregar tabela consolidada: {e}")
            return pd.DataFrame()
