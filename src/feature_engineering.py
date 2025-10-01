"""
Módulo de engenharia de features para o projeto Decision AI
Responsável pela criação de novas variáveis relevantes e transformações
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Tuple
import logging
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.decomposition import PCA
from .utils import calculate_text_similarity, normalize_score

logger = logging.getLogger(__name__)

class FeatureEngineer:
    """
    Classe responsável pela engenharia de features
    """
    
    def __init__(self):
        self.label_encoders = {}
        self.scaler = StandardScaler()
        self.tfidf_vectorizer = None
        self.pca = None
        self.feature_names = []
        
    def create_skill_compatibility_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Cria features de compatibilidade de habilidades técnicas
        
        Args:
            df: DataFrame com dados processados
            
        Returns:
            DataFrame com features de compatibilidade adicionadas
        """
        logger.info("Criando features de compatibilidade de habilidades...")
        
        df_copy = df.copy()
        
        # Features de compatibilidade de skills
        df_copy['skills_match_count'] = 0
        df_copy['skills_match_percentage'] = 0.0
        df_copy['skills_similarity_score'] = 0.0
        
        for idx, row in df_copy.iterrows():
            try:
                # Skills do candidato
                candidate_skills = str(row.get('skills_tecnicas', '')).split(', ')
                candidate_skills = [s.strip() for s in candidate_skills if s.strip()]
                
                # Skills da vaga
                vaga_skills = str(row.get('skills_requeridas', '')).split(', ')
                vaga_skills = [s.strip() for s in vaga_skills if s.strip()]
                
                if candidate_skills and vaga_skills:
                    # Conta skills em comum
                    common_skills = set(candidate_skills).intersection(set(vaga_skills))
                    df_copy.at[idx, 'skills_match_count'] = len(common_skills)
                    
                    # Percentual de match
                    df_copy.at[idx, 'skills_match_percentage'] = len(common_skills) / len(vaga_skills) if vaga_skills else 0.0
                    
                    # Score de similaridade
                    candidate_text = ' '.join(candidate_skills)
                    vaga_text = ' '.join(vaga_skills)
                    df_copy.at[idx, 'skills_similarity_score'] = calculate_text_similarity(candidate_text, vaga_text)
                
            except Exception as e:
                logger.warning(f"Erro ao processar skills para linha {idx}: {e}")
                continue
        
        return df_copy
    
    def create_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Cria features temporais adicionais
        
        Args:
            df: DataFrame com dados processados
            
        Returns:
            DataFrame com features temporais adicionadas
        """
        logger.info("Criando features temporais...")
        
        df_copy = df.copy()
        
        # Diferença entre data de candidatura e atualização
        df_copy['dias_entre_candidatura_atualizacao'] = 0
        
        for idx, row in df_copy.iterrows():
            try:
                candidatura_year = row.get('ano_candidatura', 0)
                candidatura_month = row.get('mes_candidatura', 0)
                atualizacao_year = row.get('ano_atualizacao', 0)
                atualizacao_month = row.get('mes_atualizacao', 0)
                
                if all([candidatura_year, candidatura_month, atualizacao_year, atualizacao_month]):
                    # Cálculo aproximado de dias (média de 30 dias por mês)
                    candidatura_days = candidatura_year * 365 + candidatura_month * 30
                    atualizacao_days = atualizacao_year * 365 + atualizacao_month * 30
                    df_copy.at[idx, 'dias_entre_candidatura_atualizacao'] = atualizacao_days - candidatura_days
                
            except Exception as e:
                logger.warning(f"Erro ao processar features temporais para linha {idx}: {e}")
                continue
        
        # Features de sazonalidade
        df_copy['mes_candidatura_sin'] = np.sin(2 * np.pi * df_copy['mes_candidatura'] / 12)
        df_copy['mes_candidatura_cos'] = np.cos(2 * np.pi * df_copy['mes_candidatura'] / 12)
        
        return df_copy
    
    def create_categorical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Cria features categóricas encodadas
        
        Args:
            df: DataFrame com dados processados
            
        Returns:
            DataFrame com features categóricas encodadas
        """
        logger.info("Criando features categóricas...")
        
        df_copy = df.copy()
        
        # Colunas categóricas para encoding
        categorical_columns = [
            'tipo_contratacao', 'prioridade_vaga', 'origem_vaga',
            'situacao_candidato', 'modalidade', 'area_atuacao'
        ]
        
        for col in categorical_columns:
            if col in df_copy.columns:
                try:
                    # Remove valores nulos para encoding
                    df_copy[f'{col}_encoded'] = df_copy[col].fillna('unknown')
                    
                    # Label encoding
                    le = LabelEncoder()
                    df_copy[f'{col}_encoded'] = le.fit_transform(df_copy[f'{col}_encoded'])
                    
                    # Salva o encoder para uso futuro
                    self.label_encoders[col] = le
                    
                    logger.info(f"Feature categórica '{col}' encodada com sucesso")
                    
                except Exception as e:
                    logger.warning(f"Erro ao encodar coluna '{col}': {e}")
                    continue
        
        return df_copy
    
    def create_text_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Cria features baseadas em texto usando TF-IDF
        
        Args:
            df: DataFrame com dados processados
            
        Returns:
            DataFrame com features de texto adicionadas
        """
        logger.info("Criando features de texto...")
        
        df_copy = df.copy()
        
        # Combina textos relevantes para análise
        text_columns = ['objetivo_profissional', 'area_atuacao', 'cargo_atual', 'objetivo_vaga']
        available_columns = [col for col in text_columns if col in df_copy.columns]
        
        if not available_columns:
            logger.warning("Nenhuma coluna de texto disponível para processamento")
            return df_copy
        
        # Cria coluna combinada de texto
        df_copy['texto_combinado'] = df_copy[available_columns].fillna('').agg(' '.join, axis=1)
        
        # Verifica se há texto válido sem remover as linhas ainda
        valid_text_mask = df_copy['texto_combinado'].str.strip() != ''
        
        if not valid_text_mask.any():
            logger.warning("Nenhum texto válido encontrado para processamento TF-IDF")
            df_copy = df_copy.drop('texto_combinado', axis=1)
            return df_copy
        
        # Trabalha apenas com linhas que têm texto válido
        df_with_text = df_copy[valid_text_mask].copy()
        df_without_text = df_copy[~valid_text_mask].copy()
        
        try:
            # TF-IDF Vectorization apenas nas linhas com texto
            self.tfidf_vectorizer = TfidfVectorizer(
                max_features=50,
                stop_words='english',
                ngram_range=(1, 2),
                min_df=1
            )
            
            tfidf_matrix = self.tfidf_vectorizer.fit_transform(df_with_text['texto_combinado'])
            tfidf_df = pd.DataFrame(
                tfidf_matrix.toarray(),
                columns=[f'text_feature_{i}' for i in range(tfidf_matrix.shape[1])],
                index=df_with_text.index
            )
            
            # Remove coluna de texto combinado das duas partes
            df_with_text = df_with_text.drop('texto_combinado', axis=1)
            if 'texto_combinado' in df_without_text.columns:
                df_without_text = df_without_text.drop('texto_combinado', axis=1)
            
            # Adiciona features de texto às linhas com texto
            df_with_text = pd.concat([df_with_text, tfidf_df], axis=1)
            
            # Para linhas sem texto, adiciona features de texto com valor 0
            if len(df_without_text) > 0:
                for col in tfidf_df.columns:
                    df_without_text[col] = 0.0
            
            # Combina de volta as duas partes
            df_copy = pd.concat([df_with_text, df_without_text], axis=0).sort_index()
            
            logger.info(f"Features de texto criadas: {tfidf_matrix.shape[1]} features para {len(df_with_text)} registros com texto")
            
        except Exception as e:
            logger.error(f"Erro ao criar features de texto: {e}")
            # Remove coluna de texto combinado em caso de erro
            if 'texto_combinado' in df_copy.columns:
                df_copy = df_copy.drop('texto_combinado', axis=1)
        
        return df_copy
    
    def create_interaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Cria features de interação entre variáveis
        
        Args:
            df: DataFrame com dados processados
            
        Returns:
            DataFrame com features de interação adicionadas
        """
        logger.info("Criando features de interação...")
        
        df_copy = df.copy()
        
        # Interação entre número de skills e match
        if 'num_skills' in df_copy.columns and 'skills_match_count' in df_copy.columns:
            df_copy['skills_efficiency'] = df_copy['skills_match_count'] / (df_copy['num_skills'] + 1)
        
        # Interação entre experiência e compatibilidade
        if 'tempo_experiencia' in df_copy.columns and 'skills_match_percentage' in df_copy.columns:
            # Converte tempo de experiência para numérico (aproximado)
            df_copy['tempo_experiencia_numeric'] = df_copy['tempo_experiencia'].apply(
                lambda x: self._convert_experience_to_numeric(x)
            )
            df_copy['experience_compatibility_score'] = (
                df_copy['tempo_experiencia_numeric'] * df_copy['skills_match_percentage']
            )
        
        # Score composto de compatibilidade
        compatibility_features = []
        if 'skills_match_percentage' in df_copy.columns:
            compatibility_features.append('skills_match_percentage')
        if 'skills_similarity_score' in df_copy.columns:
            compatibility_features.append('skills_similarity_score')
        
        if compatibility_features:
            df_copy['compatibility_score'] = df_copy[compatibility_features].mean(axis=1)
        
        return df_copy
    
    def _convert_experience_to_numeric(self, experience_str: str) -> float:
        """
        Converte string de experiência para valor numérico
        
        Args:
            experience_str: String com tempo de experiência
            
        Returns:
            Valor numérico em anos
        """
        if not isinstance(experience_str, str):
            return 0.0
        
        experience_str = experience_str.lower().strip()
        
        try:
            # Padrões comuns de tempo
            if 'ano' in experience_str or 'anos' in experience_str:
                # Extrai número de anos
                import re
                numbers = re.findall(r'\d+', experience_str)
                if numbers:
                    return float(numbers[0])
            
            elif 'mes' in experience_str or 'meses' in experience_str:
                # Converte meses para anos
                numbers = re.findall(r'\d+', experience_str)
                if numbers:
                    return float(numbers[0]) / 12
            
            elif 'dia' in experience_str or 'dias' in experience_str:
                # Converte dias para anos
                numbers = re.findall(r'\d+', experience_str)
                if numbers:
                    return float(numbers[0]) / 365
            
            return 0.0
            
        except Exception:
            return 0.0
    
    def normalize_numeric_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Normaliza features numéricas
        
        Args:
            df: DataFrame com dados processados
            
        Returns:
            DataFrame com features normalizadas
        """
        logger.info("Normalizando features numéricas...")
        
        df_copy = df.copy()
        
        # Identifica colunas numéricas
        numeric_columns = df_copy.select_dtypes(include=[np.number]).columns.tolist()
        
        # Remove colunas que não devem ser normalizadas
        exclude_columns = ['applicant_id', 'vaga_id', 'prospect_id', 'codigo_candidato']
        numeric_columns = [col for col in numeric_columns if col not in exclude_columns]
        
        if numeric_columns:
            try:
                # Filtra colunas que realmente existem no DataFrame
                existing_numeric_columns = [col for col in numeric_columns if col in df_copy.columns]
                
                if existing_numeric_columns:
                    # Remove colunas com apenas NaN ou apenas um valor único
                    valid_numeric_columns = []
                    for col in existing_numeric_columns:
                        if not df_copy[col].isna().all() and df_copy[col].nunique() > 1:
                            valid_numeric_columns.append(col)
                    
                    if valid_numeric_columns:
                        # Preenche NaN com mediana antes de normalizar
                        from sklearn.impute import SimpleImputer
                        imputer = SimpleImputer(strategy='median')
                        imputed_data = imputer.fit_transform(df_copy[valid_numeric_columns])
                        
                        # Aplica normalização apenas em colunas válidas
                        normalized_data = self.scaler.fit_transform(imputed_data)
                        df_copy[valid_numeric_columns] = normalized_data
                        logger.info(f"Features numéricas normalizadas: {len(valid_numeric_columns)} colunas")
                    else:
                        logger.warning("Nenhuma coluna numérica válida para normalização")
                else:
                    logger.warning("Nenhuma coluna numérica encontrada para normalização")
                
            except Exception as e:
                logger.error(f"Erro ao normalizar features numéricas: {e}")
        
        return df_copy
    
    def reduce_dimensionality(self, df: pd.DataFrame, n_components: int = 50) -> pd.DataFrame:
        """
        Reduz dimensionalidade usando PCA
        
        Args:
            df: DataFrame com dados processados
            n_components: Número de componentes para manter
            
        Returns:
            DataFrame com dimensionalidade reduzida
        """
        logger.info("Reduzindo dimensionalidade com PCA...")
        
        df_copy = df.copy()
        
        # Identifica colunas numéricas para PCA
        numeric_columns = df_copy.select_dtypes(include=[np.number]).columns.tolist()
        exclude_columns = ['applicant_id', 'vaga_id', 'prospect_id', 'codigo_candidato']
        numeric_columns = [col for col in numeric_columns if col not in exclude_columns]
        
        if len(numeric_columns) < n_components:
            logger.warning(f"PCA não aplicado: número de features ({len(numeric_columns)}) menor que n_components ({n_components})")
            return df_copy
        
        try:
            # Prepara dados para PCA, tratando valores NaN
            pca_data = df_copy[numeric_columns].copy()
            
            # Remove linhas com muitos valores NaN (>50% das colunas)
            threshold = len(numeric_columns) * 0.5
            pca_data = pca_data.dropna(thresh=threshold)
            
            if len(pca_data) == 0:
                logger.warning("PCA não aplicado: nenhum dado válido após remoção de NaN")
                return df_copy
            
            # Preenche valores NaN restantes com a mediana
            from sklearn.impute import SimpleImputer
            imputer = SimpleImputer(strategy='median')
            pca_data_imputed = imputer.fit_transform(pca_data)
            
            # Verifica se ainda há NaN ou infinitos
            if np.isnan(pca_data_imputed).any() or np.isinf(pca_data_imputed).any():
                logger.warning("PCA não aplicado: dados ainda contêm NaN ou infinitos após tratamento")
                return df_copy
            
            # Aplica PCA
            self.pca = PCA(n_components=min(n_components, len(numeric_columns), pca_data_imputed.shape[0]))
            pca_features = self.pca.fit_transform(pca_data_imputed)
            
            # Cria DataFrame com features PCA, mantendo apenas os índices válidos
            pca_df = pd.DataFrame(
                pca_features,
                columns=[f'pca_component_{i}' for i in range(pca_features.shape[1])],
                index=pca_data.index
            )
            
            # Alinha dados: mantém apenas linhas que têm dados PCA válidos
            df_copy = df_copy.loc[pca_data.index].copy()
            
            # Verifica se os índices coincidem antes de concatenar
            if len(df_copy.index.intersection(pca_df.index)) == len(pca_df):
                df_copy = pd.concat([df_copy, pca_df], axis=1)
            else:
                logger.warning("Índices não coincidem entre dados originais e PCA. Pulando concatenação PCA.")
                return df_copy
            
            logger.info(f"PCA aplicado: {pca_features.shape[1]} componentes criados para {len(pca_data)} registros")
            
        except Exception as e:
            logger.error(f"Erro ao aplicar PCA: {e}")
        
        return df_copy
    
    def run_full_feature_engineering(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Executa todo o pipeline de engenharia de features
        
        Args:
            df: DataFrame com dados pré-processados
            
        Returns:
            DataFrame com todas as features criadas
        """
        logger.info("Iniciando pipeline completo de engenharia de features...")
        logger.info(f"Shape inicial: {df.shape}")
        
        # Aplica cada etapa de engenharia de features
        df = self.create_skill_compatibility_features(df)
        logger.info(f"Após skills: {df.shape}")
        
        df = self.create_temporal_features(df)
        logger.info(f"Após temporal: {df.shape}")
        
        df = self.create_categorical_features(df)
        logger.info(f"Após categóricas: {df.shape}")
        
        df = self.create_text_features(df)
        logger.info(f"Após texto: {df.shape}")
        
        df = self.create_interaction_features(df)
        logger.info(f"Após interação: {df.shape}")
        
        df = self.normalize_numeric_features(df)
        logger.info(f"Após normalização: {df.shape}")
        
        # Reduz dimensionalidade se necessário
        if len(df.columns) > 100:
            df = self.reduce_dimensionality(df, n_components=50)
            logger.info(f"Após PCA: {df.shape}")
        
        # Verifica consistência final
        if df.index.duplicated().any():
            logger.warning("Índices duplicados detectados. Removendo duplicatas.")
            df = df[~df.index.duplicated()]
            
        # Salva nomes das features
        self.feature_names = df.columns.tolist()
        
        logger.info(f"Pipeline de engenharia de features concluído. Shape final: {df.shape}")
        
        return df
    
    def get_feature_importance_summary(self) -> Dict[str, Any]:
        """
        Retorna resumo das features criadas
        
        Returns:
            Dicionário com informações sobre as features
        """
        if not self.feature_names:
            return {"error": "Features não foram criadas"}
        
        feature_categories = {
            'text_features': [f for f in self.feature_names if f.startswith('text_feature_')],
            'pca_features': [f for f in self.feature_names if f.startswith('pca_component_')],
            'categorical_features': [f for f in self.feature_names if f.endswith('_encoded')],
            'skill_features': [f for f in self.feature_names if 'skill' in f.lower()],
            'temporal_features': [f for f in self.feature_names if any(x in f.lower() for x in ['ano', 'mes', 'dia', 'temporal'])],
            'interaction_features': [f for f in self.feature_names if any(x in f.lower() for x in ['efficiency', 'compatibility', 'interaction'])]
        }
        
        return {
            'total_features': len(self.feature_names),
            'feature_categories': feature_categories,
            'feature_names': self.feature_names
        }
