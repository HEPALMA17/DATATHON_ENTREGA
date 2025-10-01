"""
Módulo para processamento e análise de transcrições de entrevistas
"""

import os
import json
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Tuple, Any
import re
from collections import Counter
import logging

logger = logging.getLogger(__name__)

class InterviewProcessor:
    """
    Classe para processar transcrições de entrevistas e extrair análises automatizadas
    """
    
    def __init__(self):
        """Inicializa o processador de entrevistas"""
        self.technical_keywords = {
            'java': ['java', 'spring', 'hibernate', 'maven', 'gradle', 'jvm', 'jdk'],
            'python': ['python', 'django', 'flask', 'pandas', 'numpy', 'fastapi', 'pytest'],
            'sql': ['sql', 'mysql', 'postgresql', 'oracle', 'sqlite', 'query', 'database'],
            'javascript': ['javascript', 'node.js', 'react', 'angular', 'vue', 'typescript'],
            'cloud': ['aws', 'azure', 'gcp', 'docker', 'kubernetes', 'terraform'],
            'devops': ['ci/cd', 'jenkins', 'gitlab', 'github actions', 'ansible'],
            'mobile': ['android', 'ios', 'react native', 'flutter', 'swift', 'kotlin']
        }
        
        self.cultural_keywords = {
            'trabalho_equipe': ['equipe', 'colaboração', 'teamwork', 'trabalhar junto'],
            'adaptabilidade': ['adaptar', 'mudança', 'flexibilidade', 'agilidade'],
            'proatividade': ['proativo', 'iniciativa', 'sugerir', 'melhorar', 'otimizar'],
            'comunicação': ['comunicar', 'explicar', 'apresentar', 'feedback'],
            'liderança': ['liderar', 'mentor', 'coaching', 'coordenar', 'gerenciar']
        }
        
        self.motivation_keywords = {
            'interesse_vaga': ['interessado', 'apaixonado', 'motivado', 'empolgado'],
            'conhecimento_empresa': ['empresa', 'missão', 'visão', 'valores', 'cultura'],
            'crescimento': ['crescer', 'aprender', 'desenvolver', 'evoluir', 'carreira'],
            'desafio': ['desafio', 'novo', 'inovação', 'criativo', 'estimulante']
        }
        
        self.positive_sentiment = ['ótimo', 'excelente', 'bom', 'positivo', 'satisfeito', 'feliz']
        self.negative_sentiment = ['ruim', 'difícil', 'problema', 'negativo', 'insatisfeito', 'triste']
    
    def process_transcription_file(self, file_path: str) -> Dict[str, Any]:
        """
        Processa um arquivo de transcrição individual
        
        Args:
            file_path: Caminho para o arquivo de transcrição
            
        Returns:
            Dicionário com as análises extraídas
        """
        try:
            # Lê o arquivo de transcrição
            with open(file_path, 'r', encoding='utf-8') as f:
                if file_path.endswith('.json'):
                    data = json.load(f)
                    transcription_text = data.get('text', '')
                else:
                    transcription_text = f.read()
            
            # Processa a transcrição
            analysis = self.analyze_transcription(transcription_text)
            
            # Adiciona metadados
            analysis['file_path'] = file_path
            analysis['processed_at'] = datetime.now().isoformat()
            analysis['text_length'] = len(transcription_text)
            
            return analysis
            
        except Exception as e:
            logger.error(f"Erro ao processar arquivo {file_path}: {e}")
            return {}
    
    def analyze_transcription(self, text: str) -> Dict[str, Any]:
        """
        Analisa uma transcrição de entrevista
        
        Args:
            text: Texto da transcrição
            
        Returns:
            Dicionário com as análises
        """
        text_lower = text.lower()
        
        # Análise técnica
        technical_scores = self._analyze_technical_skills(text_lower)
        
        # Análise cultural
        cultural_scores = self._analyze_cultural_fit(text_lower)
        
        # Análise de motivação
        motivation_scores = self._analyze_motivation(text_lower)
        
        # Análise de sentimento
        sentiment_scores = self._analyze_sentiment(text_lower)
        
        # Análise de comunicação
        communication_scores = self._analyze_communication(text)
        
        return {
            'technical_analysis': technical_scores,
            'cultural_fit': cultural_scores,
            'motivation': motivation_scores,
            'sentiment': sentiment_scores,
            'communication': communication_scores
        }
    
    def _analyze_technical_skills(self, text: str) -> Dict[str, int]:
        """Analisa habilidades técnicas mencionadas"""
        scores = {}
        
        for skill, keywords in self.technical_keywords.items():
            mentions = sum(text.count(keyword) for keyword in keywords)
            # Normaliza o score baseado na frequência de menções
            score = min(100, mentions * 20)  # Máximo 100
            scores[f'Score_{skill.title()}'] = score
        
        return scores
    
    def _analyze_cultural_fit(self, text: str) -> Dict[str, int]:
        """Analisa fit cultural"""
        scores = {}
        
        for aspect, keywords in self.cultural_keywords.items():
            mentions = sum(text.count(keyword) for keyword in keywords)
            score = min(100, mentions * 25)  # Máximo 100
            scores[f'Score_{aspect.title()}'] = score
        
        # Calcula score geral de fit cultural
        avg_score = np.mean(list(scores.values())) if scores else 0
        scores['Score_Fit_Cultural'] = int(avg_score)
        
        return scores
    
    def _analyze_motivation(self, text: str) -> Dict[str, int]:
        """Analisa motivação e engajamento"""
        scores = {}
        
        for indicator, keywords in self.motivation_keywords.items():
            mentions = sum(text.count(keyword) for keyword in keywords)
            score = min(100, mentions * 30)  # Máximo 100
            scores[f'Score_{indicator.title()}'] = score
        
        # Calcula score geral de motivação
        avg_score = np.mean(list(scores.values())) if scores else 0
        scores['Score_Motivacao'] = int(avg_score)
        
        return scores
    
    def _analyze_sentiment(self, text: str) -> Dict[str, int]:
        """Analisa sentimento da transcrição"""
        positive_count = sum(text.count(word) for word in self.positive_sentiment)
        negative_count = sum(text.count(word) for word in self.negative_sentiment)
        
        total_sentiment = positive_count + negative_count
        
        if total_sentiment == 0:
            return {
                'Sentimento_Positivo': 50,
                'Sentimento_Negativo': 50
            }
        
        positive_ratio = positive_count / total_sentiment
        negative_ratio = negative_count / total_sentiment
        
        return {
            'Sentimento_Positivo': int(positive_ratio * 100),
            'Sentimento_Negativo': int(negative_ratio * 100)
        }
    
    def _analyze_communication(self, text: str) -> Dict[str, int]:
        """Analisa aspectos de comunicação"""
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        # Calcula métricas de comunicação
        avg_sentence_length = np.mean([len(s.split()) for s in sentences]) if sentences else 0
        
        # Clareza baseada na estrutura das frases
        clarity_score = min(100, max(0, 100 - (avg_sentence_length - 15) * 2))
        
        # Fluidez baseada na variação no tamanho das frases
        if len(sentences) > 1:
            sentence_variance = np.std([len(s.split()) for s in sentences])
            fluency_score = min(100, max(0, 100 - sentence_variance * 3))
        else:
            fluency_score = 50
        
        # Vocabulário técnico
        technical_words = sum(len(keywords) for keywords in self.technical_keywords.values())
        tech_vocab_score = min(100, (technical_words / len(text.split())) * 1000)
        
        return {
            'Score_Comunicacao': int((clarity_score + fluency_score) / 2),
            'Score_Proatividade': int(np.random.randint(70, 90)),  # Simulado
            'Score_Clareza': int(clarity_score),
            'Score_Fluidez': int(fluency_score),
            'Score_Vocabulario_Tecnico': int(tech_vocab_score)
        }
    
    def process_directory(self, directory_path: str) -> List[Dict[str, Any]]:
        """
        Processa todas as transcrições em um diretório
        
        Args:
            directory_path: Caminho para o diretório
            
        Returns:
            Lista de análises processadas
        """
        results = []
        
        if not os.path.exists(directory_path):
            logger.warning(f"Diretório {directory_path} não encontrado")
            return results
        
        for filename in os.listdir(directory_path):
            if filename.endswith(('.txt', '.json')):
                file_path = os.path.join(directory_path, filename)
                analysis = self.process_transcription_file(file_path)
                if analysis:
                    analysis['filename'] = filename
                    results.append(analysis)
        
        return results
    
    def create_standardized_table(self, interviews_data: List[Dict[str, Any]], 
                                applicants_data: List[Dict[str, Any]], 
                                prospects_data: List[Dict[str, Any]] = None) -> pd.DataFrame:
        """
        Cria a tabela candidatos_padronizados combinando dados de applicants, prospects e análises de entrevistas
        
        Args:
            interviews_data: Lista de análises de entrevistas
            applicants_data: Lista de dados de candidatos
            prospects_data: Lista de dados de prospects (opcional)
            
        Returns:
            DataFrame com candidatos padronizados
        """
        standardized_candidates = []
        
        # Converte applicants_data para DataFrame se necessário
        if isinstance(applicants_data, list):
            applicants_df = pd.DataFrame(applicants_data)
        else:
            applicants_df = applicants_data
        
        # Converte prospects_data para DataFrame se fornecido
        prospects_df = None
        if prospects_data is not None:
            if isinstance(prospects_data, list):
                prospects_df = pd.DataFrame(prospects_data)
            else:
                prospects_df = prospects_data
        
        # Processa cada candidato
        for idx, candidate in applicants_df.iterrows():
            standardized_candidate = candidate.to_dict()
            
            # Adiciona dados de prospect se disponível
            if prospects_df is not None and idx < len(prospects_df):
                prospect_data = prospects_df.iloc[idx]
                # Adiciona todas as colunas de prospect
                for col in prospect_data.index:
                    if col not in standardized_candidate:  # Evita sobrescrever dados existentes
                        standardized_candidate[f'Prospect_{col}'] = prospect_data[col]
            
            # Busca análise de entrevista correspondente (simulação)
            # Em um cenário real, você faria o match por ID ou email
            if idx < len(interviews_data):
                interview_analysis = interviews_data[idx]
                
                # Adiciona scores técnicos
                technical_scores = interview_analysis.get('technical_analysis', {})
                for score_name, score_value in technical_scores.items():
                    standardized_candidate[score_name] = score_value
                
                # Adiciona scores culturais
                cultural_scores = interview_analysis.get('cultural_fit', {})
                for score_name, score_value in cultural_scores.items():
                    standardized_candidate[score_name] = score_value
                
                # Adiciona scores de motivação
                motivation_scores = interview_analysis.get('motivation', {})
                for score_name, score_value in motivation_scores.items():
                    standardized_candidate[score_name] = score_value
                
                # Adiciona scores de sentimento
                sentiment_scores = interview_analysis.get('sentiment', {})
                for score_name, score_value in sentiment_scores.items():
                    standardized_candidate[score_name] = score_value
                
                # Adiciona scores de comunicação
                communication_scores = interview_analysis.get('communication', {})
                for score_name, score_value in communication_scores.items():
                    standardized_candidate[score_name] = score_value
                
                # Adiciona metadados da entrevista
                standardized_candidate['interview_processed'] = True
                standardized_candidate['interview_date'] = interview_analysis.get('processed_at', '')
                standardized_candidate['transcription_length'] = interview_analysis.get('text_length', 0)
                
                # Adiciona histórico de entrevistas
                standardized_candidate['Vagas_Entrevistadas'] = f"Vaga_{np.random.randint(1, 100)}"  # Simulado
                standardized_candidate['Resultado_Entrevista'] = np.random.choice(['Aprovado', 'Reprovado', 'Pendente'])
                standardized_candidate['Motivo_Nao_Aprovacao'] = np.random.choice([
                    'Não se aplica', 'Falta de experiência técnica', 'Fit cultural inadequado', 
                    'Expectativa salarial incompatível', 'Disponibilidade não adequada'
                ])
                standardized_candidate['Numero_Entrevistas'] = np.random.randint(1, 5)
                standardized_candidate['Primeira_Entrevista'] = np.random.choice([True, False])
            
            else:
                # Candidato sem entrevista processada
                for score_name in ['Score_Java', 'Score_Python', 'Score_SQL', 'Score_Fit_Cultural', 
                                 'Score_Motivacao', 'Sentimento_Positivo', 'Sentimento_Negativo',
                                 'Score_Comunicacao', 'Score_Proatividade']:
                    standardized_candidate[score_name] = None
                
                standardized_candidate['interview_processed'] = False
                standardized_candidate['interview_date'] = None
                standardized_candidate['transcription_length'] = 0
                
                # Histórico de entrevistas para candidatos sem entrevista
                standardized_candidate['Vagas_Entrevistadas'] = None
                standardized_candidate['Resultado_Entrevista'] = None
                standardized_candidate['Motivo_Nao_Aprovacao'] = None
                standardized_candidate['Numero_Entrevistas'] = 0
                standardized_candidate['Primeira_Entrevista'] = True
            
            standardized_candidates.append(standardized_candidate)
        
        return pd.DataFrame(standardized_candidates)
    
    def save_standardized_table(self, df: pd.DataFrame, output_path: str = 'candidatos_padronizados.csv'):
        """
        Salva a tabela padronizada em CSV
        
        Args:
            df: DataFrame com candidatos padronizados
            output_path: Caminho para salvar o arquivo
        """
        try:
            df.to_csv(output_path, index=False, encoding='utf-8')
            logger.info(f"Tabela padronizada salva em {output_path}")
            return True
        except Exception as e:
            logger.error(f"Erro ao salvar tabela padronizada: {e}")
            return False
    
    def load_standardized_table(self, file_path: str = 'candidatos_padronizados.csv') -> pd.DataFrame:
        """
        Carrega a tabela padronizada de um arquivo CSV
        
        Args:
            file_path: Caminho para o arquivo CSV
            
        Returns:
            DataFrame com candidatos padronizados
        """
        try:
            if os.path.exists(file_path):
                return pd.read_csv(file_path, encoding='utf-8')
            else:
                logger.warning(f"Arquivo {file_path} não encontrado")
                return pd.DataFrame()
        except Exception as e:
            logger.error(f"Erro ao carregar tabela padronizada: {e}")
            return pd.DataFrame()

