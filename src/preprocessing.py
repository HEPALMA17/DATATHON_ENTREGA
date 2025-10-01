"""
Módulo de pré-processamento de dados para o projeto Decision AI
Responsável pela limpeza, tratamento e preparação dos dados
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Tuple
import logging
from .utils import (
    load_json_data, 
    clean_text, 
    extract_skills_from_text,
    create_date_features,
    safe_get_nested,
    validate_data_quality
)

logger = logging.getLogger(__name__)

# Lista de habilidades técnicas comuns em TI
TECH_SKILLS = [
    'python', 'java', 'javascript', 'typescript', 'react', 'angular', 'vue',
    'node.js', 'express', 'django', 'flask', 'spring', 'hibernate',
    'sql', 'mysql', 'postgresql', 'mongodb', 'redis', 'elasticsearch',
    'docker', 'kubernetes', 'aws', 'azure', 'gcp', 'terraform',
    'git', 'jenkins', 'ci/cd', 'agile', 'scrum', 'kanban',
    'machine learning', 'ai', 'data science', 'big data', 'spark',
    'html', 'css', 'bootstrap', 'sass', 'less', 'webpack',
    'linux', 'unix', 'shell', 'bash', 'powershell',
    'rest api', 'graphql', 'microservices', 'serverless'
]

class DataPreprocessor:
    """
    Classe responsável pelo pré-processamento dos dados
    """
    
    def __init__(self):
        self.applicants_data = {}
        self.vagas_data = {}
        self.prospects_data = {}
        self.processed_applicants = None
        self.processed_vagas = None
        self.processed_prospects = None
        
    def load_all_data(self, base_path: str = ".") -> bool:
        """
        Carrega todos os arquivos de dados
        
        Args:
            base_path: Caminho base para os arquivos
            
        Returns:
            True se todos os dados foram carregados com sucesso
        """
        try:
            logger.info("Carregando dados dos arquivos JSON...")
            
            self.applicants_data = load_json_data(f"{base_path}/applicants.json")
            self.vagas_data = load_json_data(f"{base_path}/vagas.json")
            self.prospects_data = load_json_data(f"{base_path}/prospects.json")
            
            if not all([self.applicants_data, self.vagas_data, self.prospects_data]):
                logger.error("Falha ao carregar um ou mais arquivos de dados")
                return False
            
            logger.info("Todos os dados foram carregados com sucesso")
            return True
            
        except Exception as e:
            logger.error(f"Erro ao carregar dados: {e}")
            return False
    
    def preprocess_applicants(self) -> pd.DataFrame:
        """
        Pré-processa os dados dos candidatos
        
        Returns:
            DataFrame com candidatos processados
        """
        logger.info("Iniciando pré-processamento dos candidatos...")
        
        processed_data = []
        
        for applicant_id, applicant_info in self.applicants_data.items():
            try:
                # Informações básicas
                basic_info = safe_get_nested(applicant_info, ['infos_basicas'], {})
                personal_info = safe_get_nested(applicant_info, ['informacoes_pessoais'], {})
                professional_info = safe_get_nested(applicant_info, ['informacoes_profissionais'], {})
                
                # Dados básicos
                applicant_record = {
                    'applicant_id': applicant_id,
                    'nome': safe_get_nested(basic_info, ['nome'], ''),
                    'email': safe_get_nested(basic_info, ['email'], ''),
                    'telefone': safe_get_nested(basic_info, ['telefone'], ''),
                    'local': safe_get_nested(basic_info, ['local'], ''),
                    'objetivo_profissional': safe_get_nested(basic_info, ['objetivo_profissional'], ''),
                    'data_atualizacao': safe_get_nested(basic_info, ['data_atualizacao'], ''),
                    'inserido_por': safe_get_nested(basic_info, ['inserido_por'], ''),
                    
                    # Informações pessoais
                    'cpf': safe_get_nested(personal_info, ['cpf'], ''),
                    'fonte_indicacao': safe_get_nested(personal_info, ['fonte_indicacao'], ''),
                    'data_aceite': safe_get_nested(personal_info, ['data_aceite'], ''),
                    
                    # Informações profissionais
                    'area_atuacao': safe_get_nested(professional_info, ['area_atuacao'], ''),
                    'cargo_atual': safe_get_nested(professional_info, ['cargo_atual'], ''),
                    'empresa_atual': safe_get_nested(professional_info, ['empresa_atual'], ''),
                    'tempo_experiencia': safe_get_nested(professional_info, ['tempo_experiencia'], ''),
                    'pretensao_salarial': safe_get_nested(professional_info, ['pretensao_salarial'], ''),
                    'disponibilidade': safe_get_nested(professional_info, ['disponibilidade'], ''),
                }
                
                # Limpeza de texto
                for key in ['nome', 'objetivo_profissional', 'area_atuacao', 'cargo_atual']:
                    if applicant_record[key]:
                        applicant_record[key] = clean_text(applicant_record[key])
                
                # Extração de habilidades técnicas
                text_for_skills = ' '.join([
                    str(applicant_record.get('objetivo_profissional', '')),
                    str(applicant_record.get('area_atuacao', '')),
                    str(applicant_record.get('cargo_atual', ''))
                ])
                
                skills_found = extract_skills_from_text(text_for_skills, TECH_SKILLS)
                applicant_record['skills_tecnicas'] = ', '.join(skills_found)
                applicant_record['num_skills'] = len(skills_found)
                
                # Features de data
                date_features = create_date_features(applicant_record['data_atualizacao'])
                applicant_record.update(date_features)
                
                processed_data.append(applicant_record)
                
            except Exception as e:
                logger.warning(f"Erro ao processar candidato {applicant_id}: {e}")
                continue
        
        self.processed_applicants = pd.DataFrame(processed_data)
        logger.info(f"Pré-processamento concluído. {len(self.processed_applicants)} candidatos processados")
        
        return self.processed_applicants
    
    def preprocess_vagas(self) -> pd.DataFrame:
        """
        Pré-processa os dados das vagas
        
        Returns:
            DataFrame com vagas processadas
        """
        logger.info("Iniciando pré-processamento das vagas...")
        
        processed_data = []
        
        for vaga_id, vaga_info in self.vagas_data.items():
            try:
                # Informações básicas da vaga
                basic_info = safe_get_nested(vaga_info, ['informacoes_basicas'], {})
                
                vaga_record = {
                    'vaga_id': vaga_id,
                    'titulo_vaga': safe_get_nested(basic_info, ['titulo_vaga'], ''),
                    'tipo_contratacao': safe_get_nested(basic_info, ['tipo_contratacao'], ''),
                    'prioridade_vaga': safe_get_nested(basic_info, ['prioridade_vaga'], ''),
                    'origem_vaga': safe_get_nested(basic_info, ['origem_vaga'], ''),
                    'requisitante': safe_get_nested(basic_info, ['requisitante'], ''),
                    'analista_responsavel': safe_get_nested(basic_info, ['analista_responsavel'], ''),
                    'objetivo_vaga': safe_get_nested(basic_info, ['objetivo_vaga'], ''),
                    'prazo_contratacao': safe_get_nested(basic_info, ['prazo_contratacao'], ''),
                }
                
                # Limpeza de texto
                for key in ['titulo_vaga', 'objetivo_vaga']:
                    if vaga_record[key]:
                        vaga_record[key] = clean_text(vaga_record[key])
                
                # Extração de habilidades técnicas da vaga
                text_for_skills = ' '.join([
                    str(vaga_record.get('titulo_vaga', '')),
                    str(vaga_record.get('objetivo_vaga', ''))
                ])
                
                skills_found = extract_skills_from_text(text_for_skills, TECH_SKILLS)
                vaga_record['skills_requeridas'] = ', '.join(skills_found)
                vaga_record['num_skills_requeridas'] = len(skills_found)
                
                # Features de data
                date_features = create_date_features(vaga_record['prazo_contratacao'])
                vaga_record.update(date_features)
                
                processed_data.append(vaga_record)
                
            except Exception as e:
                logger.warning(f"Erro ao processar vaga {vaga_id}: {e}")
                continue
        
        self.processed_vagas = pd.DataFrame(processed_data)
        logger.info(f"Pré-processamento concluído. {len(self.processed_vagas)} vagas processadas")
        
        return self.processed_vagas
    
    def preprocess_prospects(self) -> pd.DataFrame:
        """
        Pré-processa os dados dos prospects
        
        Returns:
            DataFrame com prospects processados
        """
        logger.info("Iniciando pré-processamento dos prospects...")
        
        processed_data = []
        
        for prospect_id, prospect_info in self.prospects_data.items():
            try:
                # Informações básicas do prospect
                titulo = safe_get_nested(prospect_info, ['titulo'], '')
                modalidade = safe_get_nested(prospect_info, ['modalidade'], '')
                prospects_list = safe_get_nested(prospect_info, ['prospects'], [])
                
                # Processa cada prospect individual
                for prospect in prospects_list:
                    # Trata erro de digitação no JSON (situacao_candidado vs situacao_candidato)
                    situacao = safe_get_nested(prospect, ['situacao_candidato'], '')
                    if not situacao:
                        situacao = safe_get_nested(prospect, ['situacao_candidado'], '')
                    
                    prospect_record = {
                        'prospect_id': prospect_id,
                        'titulo_vaga': titulo,
                        'modalidade': modalidade,
                        'nome_candidato': safe_get_nested(prospect, ['nome'], ''),
                        'codigo_candidato': safe_get_nested(prospect, ['codigo'], ''),
                        'situacao_candidato': situacao,
                        'data_candidatura': safe_get_nested(prospect, ['data_candidatura'], ''),
                        'ultima_atualizacao': safe_get_nested(prospect, ['ultima_atualizacao'], ''),
                        'comentario': safe_get_nested(prospect, ['comentario'], ''),
                        'recrutador': safe_get_nested(prospect, ['recrutador'], ''),
                    }
                    
                    # Limpeza de texto
                    for key in ['nome_candidato', 'comentario', 'recrutador']:
                        if prospect_record[key]:
                            prospect_record[key] = clean_text(prospect_record[key])
                    
                    # Features de data
                    candidatura_features = create_date_features(prospect_record['data_candidatura'])
                    atualizacao_features = create_date_features(prospect_record['ultima_atualizacao'])
                    
                    prospect_record.update({
                        'ano_candidatura': candidatura_features['year'],
                        'mes_candidatura': candidatura_features['month'],
                        'ano_atualizacao': atualizacao_features['year'],
                        'mes_atualizacao': atualizacao_features['month']
                    })
                    
                    processed_data.append(prospect_record)
                
            except Exception as e:
                logger.warning(f"Erro ao processar prospect {prospect_id}: {e}")
                continue
        
        self.processed_prospects = pd.DataFrame(processed_data)
        logger.info(f"Pré-processamento concluído. {len(self.processed_prospects)} prospects processados")
        
        return self.processed_prospects
    
    def create_merged_dataset(self) -> pd.DataFrame:
        """
        Cria dataset unificado combinando candidatos, vagas e prospects
        
        Returns:
            DataFrame unificado
        """
        if (self.processed_applicants is None or 
            self.processed_vagas is None or 
            self.processed_prospects is None):
            logger.error("Dados não foram pré-processados. Execute o pré-processamento primeiro.")
            return pd.DataFrame()
        
        logger.info("Criando dataset unificado...")
        
        # Merge prospects com vagas
        prospects_vagas = pd.merge(
            self.processed_prospects,
            self.processed_vagas,
            left_on='titulo_vaga',
            right_on='titulo_vaga',
            how='left',
            suffixes=('_prospect', '_vaga')
        )
        
        # Merge com candidatos
        final_dataset = pd.merge(
            prospects_vagas,
            self.processed_applicants,
            left_on='codigo_candidato',
            right_on='applicant_id',
            how='left',
            suffixes=('_prospect', '_applicant')
        )
        
        logger.info(f"Dataset unificado criado com {len(final_dataset)} registros")
        return final_dataset
    
    def run_full_preprocessing(self, base_path: str = ".") -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Executa todo o pipeline de pré-processamento
        
        Args:
            base_path: Caminho base para os arquivos
            
        Returns:
            Tupla com (applicants, vagas, prospects, dataset_unificado)
        """
        logger.info("Iniciando pipeline completo de pré-processamento...")
        
        # Carrega dados
        if not self.load_all_data(base_path):
            raise RuntimeError("Falha ao carregar dados")
        
        # Pré-processa cada dataset
        applicants = self.preprocess_applicants()
        vagas = self.preprocess_vagas()
        prospects = self.preprocess_prospects()
        
        # Cria dataset unificado
        merged_dataset = self.create_merged_dataset()
        
        logger.info("Pipeline de pré-processamento concluído com sucesso!")
        
        return applicants, vagas, prospects, merged_dataset
    
    def get_data_quality_report(self) -> Dict[str, Any]:
        """
        Gera relatório de qualidade dos dados processados
        
        Returns:
            Dicionário com métricas de qualidade
        """
        if self.processed_applicants is None:
            return {"error": "Dados não foram processados"}
        
        quality_report = {
            'applicants': validate_data_quality(self.processed_applicants),
            'vagas': validate_data_quality(self.processed_vagas) if self.processed_vagas is not None else {},
            'prospects': validate_data_quality(self.processed_prospects) if self.processed_prospects is not None else {}
        }
        
        return quality_report
