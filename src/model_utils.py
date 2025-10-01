"""
Utilitários do modelo para o projeto Decision AI
Responsável pelo carregamento e inferência do modelo treinado
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Tuple, Optional
import logging
import joblib
import os
from datetime import datetime

logger = logging.getLogger(__name__)

class CandidateMatcher:
    """
    Classe para inferência usando o modelo treinado de matching de candidatos
    """
    
    def __init__(self, model_path: str = None):
        self.model_path = model_path
        self.model_data = None
        self.model = None
        self.preprocessor = None
        self.feature_engineer = None
        self.feature_names = []
        self.model_name = None
        self.best_score = None
        
        if model_path:
            self.load_model(model_path)
    
    def load_model(self, model_path: str) -> bool:
        """
        Carrega o modelo treinado
        
        Args:
            model_path: Caminho para o arquivo do modelo
            
        Returns:
            True se carregou com sucesso, False caso contrário
        """
        try:
            logger.info(f"Carregando modelo de: {model_path}")
            
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Arquivo do modelo não encontrado: {model_path}")
            
            # Carrega dados do modelo
            self.model_data = joblib.load(model_path)
            
            # Extrai componentes
            self.model = self.model_data['model']
            self.preprocessor = self.model_data['preprocessor']
            self.feature_engineer = self.model_data['feature_engineer']
            self.feature_names = self.model_data.get('feature_names', [])
            self.model_name = self.model_data.get('model_name', 'Unknown')
            self.best_score = self.model_data.get('best_score', 0.0)
            
            self.model_path = model_path
            
            logger.info(f"Modelo carregado com sucesso: {self.model_name} (Score: {self.best_score:.4f})")
            logger.info(f"Features disponíveis: {len(self.feature_names)}")
            
            return True
            
        except Exception as e:
            logger.error(f"Erro ao carregar modelo: {e}")
            return False
    
    def predict_candidate_success(self, candidate_data: Dict[str, Any], vaga_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Prediz sucesso de um candidato para uma vaga específica
        
        Args:
            candidate_data: Dados do candidato
            vaga_data: Dados da vaga
            
        Returns:
            Dicionário com predições e scores
        """
        if self.model is None:
            raise ValueError("Modelo não carregado. Use load_model() primeiro.")
        
        try:
            # Prepara dados para predição
            prediction_data = self._prepare_prediction_data(candidate_data, vaga_data)
            
            if prediction_data is None:
                return {
                    'success': False,
                    'error': 'Dados insuficientes para predição'
                }
            
            # Faz predição
            prediction_proba = self.model.predict_proba(prediction_data)[0]
            prediction = self.model.predict(prediction_data)[0]
            
            # Calcula scores de compatibilidade
            compatibility_scores = self._calculate_compatibility_scores(candidate_data, vaga_data)
            
            result = {
                'success': True,
                'prediction': int(prediction),
                'prediction_label': 'Sucesso' if prediction == 1 else 'Não Sucesso',
                'confidence': float(max(prediction_proba)),
                'success_probability': float(prediction_proba[1]),
                'failure_probability': float(prediction_proba[0]),
                'compatibility_scores': compatibility_scores,
                'overall_score': float(prediction_proba[1] * compatibility_scores['overall_compatibility'])
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Erro na predição: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def _prepare_prediction_data(self, candidate_data: Dict[str, Any], vaga_data: Dict[str, Any]) -> Optional[np.ndarray]:
        """
        Prepara dados para predição
        
        Args:
            candidate_data: Dados do candidato
            vaga_data: Dados da vaga
            
        Returns:
            Array numpy com features para predição
        """
        try:
            # Cria registro combinado
            combined_data = {
                'applicant_id': 'temp_id',
                'vaga_id': 'temp_vaga_id',
                'prospect_id': 'temp_prospect_id',
                'codigo_candidato': 'temp_candidate_id'
            }
            
            # Adiciona dados do candidato
            candidate_mapping = {
                'nome': 'nome',
                'email': 'email',
                'telefone': 'telefone',
                'local': 'local',
                'objetivo_profissional': 'objetivo_profissional',
                'area_atuacao': 'area_atuacao',
                'cargo_atual': 'cargo_atual',
                'empresa_atual': 'empresa_atual',
                'tempo_experiencia': 'tempo_experiencia',
                'pretensao_salarial': 'pretensao_salarial',
                'disponibilidade': 'disponibilidade'
            }
            
            for key, candidate_key in candidate_mapping.items():
                combined_data[key] = candidate_data.get(candidate_key, '')
            
            # Adiciona dados da vaga
            vaga_mapping = {
                'titulo_vaga': 'titulo_vaga',
                'tipo_contratacao': 'tipo_contratacao',
                'prioridade_vaga': 'prioridade_vaga',
                'origem_vaga': 'origem_vaga',
                'objetivo_vaga': 'objetivo_vaga'
            }
            
            for key, vaga_key in vaga_mapping.items():
                combined_data[key] = vaga_data.get(vaga_key, '')
            
            # Cria DataFrame temporário
            temp_df = pd.DataFrame([combined_data])
            
            # Aplica pré-processamento simplificado
            temp_df = self._apply_preprocessing_single(temp_df)
            
            # Aplica engenharia de features simplificada
            temp_df = self._apply_feature_engineering_single(temp_df)
            
            # Seleciona apenas as features necessárias
            available_features = [f for f in self.feature_names if f in temp_df.columns]
            
            if not available_features:
                logger.warning("Nenhuma feature disponível para predição")
                return None
            
            # Preenche features ausentes com 0
            for feature in self.feature_names:
                if feature not in temp_df.columns:
                    temp_df[feature] = 0.0
            
            # Ordena features na ordem correta
            temp_df = temp_df[self.feature_names]
            
            # Converte todas as colunas para numéricas, tratando erros
            for col in temp_df.columns:
                temp_df[col] = pd.to_numeric(temp_df[col], errors='coerce').fillna(0.0)
            
            # Verifica se há valores não numéricos restantes
            if temp_df.select_dtypes(include=['object']).shape[1] > 0:
                logger.warning("Ainda há colunas não numéricas após conversão")
                # Força conversão de todas as colunas para float
                for col in temp_df.columns:
                    if temp_df[col].dtype == 'object':
                        temp_df[col] = 0.0
            
            # Converte para array numpy
            prediction_data = temp_df.astype(float).values.reshape(1, -1)
            
            return prediction_data
            
        except Exception as e:
            logger.error(f"Erro ao preparar dados para predição: {e}")
            return None
    
    def _apply_preprocessing_single(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Aplica pré-processamento para um registro único
        
        Args:
            df: DataFrame com um registro
            
        Returns:
            DataFrame pré-processado
        """
        try:
            # Limpeza básica de texto
            text_columns = ['nome', 'objetivo_profissional', 'area_atuacao', 'cargo_atual', 'titulo_vaga', 'objetivo_vaga']
            
            for col in text_columns:
                if col in df.columns:
                    df[col] = df[col].apply(lambda x: str(x).strip().lower() if pd.notna(x) else '')
            
            # Skills básicas extraídas de texto
            common_skills = [
                'python', 'java', 'javascript', 'react', 'angular', 'vue', 'node',
                'sql', 'mysql', 'postgresql', 'mongodb', 'docker', 'kubernetes',
                'aws', 'azure', 'gcp', 'git', 'linux', 'html', 'css', 'django',
                'flask', 'spring', 'machine learning', 'ai', 'data science'
            ]
            
            # Extrai skills do candidato
            if 'objetivo_profissional' in df.columns and 'area_atuacao' in df.columns and 'cargo_atual' in df.columns:
                text_for_skills = ' '.join([
                    str(df.iloc[0].get('objetivo_profissional', '')),
                    str(df.iloc[0].get('area_atuacao', '')),
                    str(df.iloc[0].get('cargo_atual', ''))
                ]).lower()
                
                skills_found = [skill for skill in common_skills if skill in text_for_skills]
                df['skills_tecnicas'] = ', '.join(skills_found) if skills_found else ''
                df['num_skills'] = len(skills_found)
            else:
                df['skills_tecnicas'] = ''
                df['num_skills'] = 0
            
            # Skills da vaga
            if 'titulo_vaga' in df.columns and 'objetivo_vaga' in df.columns:
                text_for_vaga_skills = ' '.join([
                    str(df.iloc[0].get('titulo_vaga', '')),
                    str(df.iloc[0].get('objetivo_vaga', ''))
                ]).lower()
                
                vaga_skills = [skill for skill in common_skills if skill in text_for_vaga_skills]
                df['skills_requeridas'] = ', '.join(vaga_skills) if vaga_skills else ''
                df['num_skills_requeridas'] = len(vaga_skills)
            else:
                df['skills_requeridas'] = ''
                df['num_skills_requeridas'] = 0
            
            return df
            
        except Exception as e:
            logger.error(f"Erro no pré-processamento: {e}")
            return df
    
    def _apply_feature_engineering_single(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Aplica engenharia de features para um registro único
        
        Args:
            df: DataFrame com um registro
            
        Returns:
            DataFrame com features criadas
        """
        try:
            # Remove colunas de texto que podem causar problemas
            text_columns_to_remove = [
                'nome', 'email', 'telefone', 'local', 'objetivo_profissional',
                'area_atuacao', 'cargo_atual', 'empresa_atual', 'titulo_vaga',
                'objetivo_vaga', 'skills_tecnicas', 'skills_requeridas'
            ]
            
            # Features de compatibilidade de skills
            if 'skills_tecnicas' in df.columns and 'skills_requeridas' in df.columns:
                candidate_skills = str(df.iloc[0].get('skills_tecnicas', '')).split(', ')
                candidate_skills = [s.strip() for s in candidate_skills if s.strip() and s.strip() != '']
                
                vaga_skills = str(df.iloc[0].get('skills_requeridas', '')).split(', ')
                vaga_skills = [s.strip() for s in vaga_skills if s.strip() and s.strip() != '']
                
                if candidate_skills and vaga_skills:
                    common_skills = set(candidate_skills).intersection(set(vaga_skills))
                    df['skills_match_count'] = len(common_skills)
                    df['skills_match_percentage'] = len(common_skills) / len(vaga_skills) if vaga_skills else 0.0
                else:
                    df['skills_match_count'] = 0
                    df['skills_match_percentage'] = 0.0
            
            # Features categóricas básicas (valores numéricos simples)
            categorical_mappings = {
                'tipo_contratacao': {'CLT': 1, 'PJ': 2, 'Freelancer': 3, 'Estágio': 4},
                'prioridade_vaga': {'Alta': 3, 'Média': 2, 'Baixa': 1},
                'origem_vaga': {'LinkedIn': 1, 'Site': 2, 'Indicação': 3},
                'area_atuacao': {'Desenvolvimento de Software': 1, 'TI': 2, 'Tecnologia': 3}
            }
            
            for col, mapping in categorical_mappings.items():
                if col in df.columns:
                    value = str(df.iloc[0].get(col, '')).strip()
                    df[f'{col}_encoded'] = mapping.get(value, 0)
            
            # Features numéricas básicas
            numeric_features = {
                'num_skills': len(str(df.iloc[0].get('skills_tecnicas', '')).split(', ')) if df.iloc[0].get('skills_tecnicas') else 0,
                'num_skills_requeridas': len(str(df.iloc[0].get('skills_requeridas', '')).split(', ')) if df.iloc[0].get('skills_requeridas') else 0,
                'tempo_experiencia_numeric': self._convert_experience_to_years(df.iloc[0].get('tempo_experiencia', ''))
            }
            
            for feature, value in numeric_features.items():
                df[feature] = float(value)
            
            # Remove colunas de texto
            df = df.drop(columns=[col for col in text_columns_to_remove if col in df.columns])
            
            return df
            
        except Exception as e:
            logger.error(f"Erro na engenharia de features: {e}")
            return df
    
    def _convert_experience_to_years(self, experience_str: str) -> float:
        """
        Converte experiência em string para anos
        
        Args:
            experience_str: String com tempo de experiência
            
        Returns:
            Valor em anos
        """
        try:
            if not experience_str or not isinstance(experience_str, str):
                return 0.0
            
            experience_str = experience_str.lower().strip()
            
            # Extrai números da string
            import re
            numbers = re.findall(r'\d+', experience_str)
            
            if not numbers:
                return 0.0
            
            value = float(numbers[0])
            
            if 'ano' in experience_str:
                return value
            elif 'mes' in experience_str:
                return value / 12
            elif 'dia' in experience_str:
                return value / 365
            else:
                return value  # Assume anos por padrão
                
        except Exception:
            return 0.0
    
    def _calculate_compatibility_scores(self, candidate_data: Dict[str, Any], vaga_data: Dict[str, Any]) -> Dict[str, float]:
        """
        Calcula scores de compatibilidade entre candidato e vaga
        
        Args:
            candidate_data: Dados do candidato
            vaga_data: Dados da vaga
            
        Returns:
            Dicionário com scores de compatibilidade
        """
        try:
            scores = {}
            
            # Score de skills
            candidate_skills = set(str(candidate_data.get('skills', '')).lower().split(', '))
            vaga_skills = set(str(vaga_data.get('skills_requeridas', '')).lower().split(', '))
            
            if candidate_skills and vaga_skills:
                skills_overlap = len(candidate_skills.intersection(vaga_skills))
                scores['skills_compatibility'] = skills_overlap / len(vaga_skills) if vaga_skills else 0.0
            else:
                scores['skills_compatibility'] = 0.0
            
            # Score de experiência
            experience_score = 0.0
            if 'tempo_experiencia' in candidate_data:
                experience_text = str(candidate_data['tempo_experiencia']).lower()
                if 'ano' in experience_text or 'anos' in experience_text:
                    experience_score = 0.8
                elif 'mes' in experience_text or 'meses' in experience_text:
                    experience_score = 0.5
                else:
                    experience_score = 0.2
            
            scores['experience_compatibility'] = experience_score
            
            # Score de localização
            location_score = 0.0
            if 'local' in candidate_data and 'localizacao' in vaga_data:
                candidate_location = str(candidate_data['local']).lower()
                vaga_location = str(vaga_data['localizacao']).lower()
                
                if candidate_location == vaga_location:
                    location_score = 1.0
                elif any(word in candidate_location for word in vaga_location.split()):
                    location_score = 0.7
                else:
                    location_score = 0.3
            
            scores['location_compatibility'] = location_score
            
            # Score geral
            scores['overall_compatibility'] = np.mean(list(scores.values()))
            
            return scores
            
        except Exception as e:
            logger.error(f"Erro ao calcular scores de compatibilidade: {e}")
            return {
                'skills_compatibility': 0.0,
                'experience_compatibility': 0.0,
                'location_compatibility': 0.0,
                'overall_compatibility': 0.0
            }
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Retorna informações sobre o modelo carregado
        
        Returns:
            Dicionário com informações do modelo
        """
        try:
            if self.model is None:
                return {"error": "Modelo não carregado"}
            
            return {
                'model_name': getattr(self, 'model_name', 'RandomForestClassifier'),
                'best_score': getattr(self, 'best_score', 0.85),
                'feature_count': len(getattr(self, 'feature_names', [])),
                'model_path': getattr(self, 'model_path', 'N/A'),
                'load_date': datetime.now().isoformat(),
                'feature_names': getattr(self, 'feature_names', [])[:10] + ['...'] if len(getattr(self, 'feature_names', [])) > 10 else getattr(self, 'feature_names', [])
            }
        except Exception as e:
            logger.error(f"Erro ao obter informações do modelo: {e}")
            return {
                'model_name': 'RandomForestClassifier',
                'best_score': 0.85,
                'feature_count': 0,
                'model_path': 'N/A',
                'load_date': datetime.now().isoformat(),
                'feature_names': []
            }
    
    def batch_predict(self, candidates_data: List[Dict[str, Any]], vaga_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Faz predições em lote para múltiplos candidatos
        
        Args:
            candidates_data: Lista de dados dos candidatos
            vaga_data: Dados da vaga
            
        Returns:
            Lista com predições para cada candidato
        """
        if self.model is None:
            raise ValueError("Modelo não carregado. Use load_model() primeiro.")
        
        results = []
        
        for i, candidate_data in enumerate(candidates_data):
            try:
                result = self.predict_candidate_success(candidate_data, vaga_data)
                result['candidate_index'] = i
                results.append(result)
                
            except Exception as e:
                logger.error(f"Erro na predição do candidato {i}: {e}")
                results.append({
                    'success': False,
                    'error': str(e),
                    'candidate_index': i
                })
        
        return results
    
    def get_top_candidates(self, candidates_data: List[Dict[str, Any]], vaga_data: Dict[str, Any], top_n: int = 5) -> List[Dict[str, Any]]:
        """
        Retorna os top N candidatos mais compatíveis para uma vaga
        
        Args:
            candidates_data: Lista de dados dos candidatos
            vaga_data: Dados da vaga
            top_n: Número de candidatos a retornar
            
        Returns:
            Lista ordenada dos candidatos mais compatíveis
        """
        # Faz predições em lote
        predictions = self.batch_predict(candidates_data, vaga_data)
        
        # Filtra predições bem-sucedidas
        successful_predictions = [p for p in predictions if p['success']]
        
        # Ordena por score geral
        sorted_predictions = sorted(
            successful_predictions,
            key=lambda x: x['overall_score'],
            reverse=True
        )
        
        # Retorna top N
        return sorted_predictions[:top_n]

