"""
Utilitários para o projeto Decision AI
Funções auxiliares reutilizáveis
"""

import json
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Tuple
import logging
from datetime import datetime
import re

# Configuração de logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_json_data(file_path: str) -> Dict[str, Any]:
    """
    Carrega dados de um arquivo JSON
    
    Args:
        file_path: Caminho para o arquivo JSON
        
    Returns:
        Dicionário com os dados carregados
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
        logger.info(f"Dados carregados com sucesso de {file_path}")
        return data
    except Exception as e:
        logger.error(f"Erro ao carregar dados de {file_path}: {e}")
        return {}

def save_json_data(data: Dict[str, Any], file_path: str) -> bool:
    """
    Salva dados em um arquivo JSON
    
    Args:
        data: Dados a serem salvos
        file_path: Caminho para o arquivo de destino
        
    Returns:
        True se salvou com sucesso, False caso contrário
    """
    try:
        with open(file_path, 'w', encoding='utf-8') as file:
            json.dump(data, file, ensure_ascii=False, indent=2)
        logger.info(f"Dados salvos com sucesso em {file_path}")
        return True
    except Exception as e:
        logger.error(f"Erro ao salvar dados em {file_path}: {e}")
        return False

def clean_text(text: str) -> str:
    """
    Limpa e normaliza texto
    
    Args:
        text: Texto a ser limpo
        
    Returns:
        Texto limpo e normalizado
    """
    if not isinstance(text, str):
        return ""
    
    # Remove caracteres especiais e normaliza espaços
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    text = text.strip().lower()
    
    return text

def extract_skills_from_text(text: str, skill_keywords: List[str]) -> List[str]:
    """
    Extrai habilidades técnicas de um texto
    
    Args:
        text: Texto para análise
        skill_keywords: Lista de palavras-chave de habilidades
        
    Returns:
        Lista de habilidades encontradas
    """
    if not text:
        return []
    
    text_clean = clean_text(text)
    found_skills = []
    
    for skill in skill_keywords:
        if skill.lower() in text_clean:
            found_skills.append(skill)
    
    return list(set(found_skills))

def calculate_text_similarity(text1: str, text2: str) -> float:
    """
    Calcula similaridade entre dois textos usando Jaccard
    
    Args:
        text1: Primeiro texto
        text2: Segundo texto
        
    Returns:
        Score de similaridade entre 0 e 1
    """
    if not text1 or not text2:
        return 0.0
    
    words1 = set(clean_text(text1).split())
    words2 = set(clean_text(text2).split())
    
    if not words1 or not words2:
        return 0.0
    
    intersection = len(words1.intersection(words2))
    union = len(words1.union(words2))
    
    return intersection / union if union > 0 else 0.0

def normalize_score(score: float, min_val: float = 0.0, max_val: float = 1.0) -> float:
    """
    Normaliza um score para o intervalo [0, 1]
    
    Args:
        score: Score a ser normalizado
        min_val: Valor mínimo esperado
        max_val: Valor máximo esperado
        
    Returns:
        Score normalizado
    """
    if max_val == min_val:
        return 0.0
    
    normalized = (score - min_val) / (max_val - min_val)
    return max(0.0, min(1.0, normalized))

def create_date_features(date_str: str) -> Dict[str, Any]:
    """
    Cria features temporais a partir de uma string de data
    
    Args:
        date_str: String de data no formato DD-MM-YYYY
        
    Returns:
        Dicionário com features temporais
    """
    try:
        # Verifica se é uma string válida
        if not date_str or not isinstance(date_str, str) or date_str.strip() == "":
            return {
                'year': 0,
                'month': 0,
                'day': 0,
                'day_of_week': 0,
                'is_weekend': False
            }
        
        # Limpa a string
        date_str = date_str.strip()
        
        # Trata casos especiais como 'Indeterminado', 'Determinado', etc.
        if date_str.lower() in ['indeterminado', 'determinado', 'n/a', 'na', 'nan', 'null', 'none']:
            return {
                'year': 0,
                'month': 0,
                'day': 0,
                'day_of_week': 0,
                'is_weekend': False
            }
        
        # Tenta diferentes formatos de data
        date_formats = ['%d-%m-%Y', '%d/%m/%Y', '%Y-%m-%d', '%d-%m-%y', '%d/%m/%y']
        
        for date_format in date_formats:
            try:
                date_obj = datetime.strptime(date_str, date_format)
                return {
                    'year': date_obj.year,
                    'month': date_obj.month,
                    'day': date_obj.day,
                    'day_of_week': date_obj.weekday(),
                    'is_weekend': date_obj.weekday() >= 5
                }
            except ValueError:
                continue
        
        # Se nenhum formato funcionou, retorna valores padrão
        logger.debug(f"Data não reconhecida '{date_str}', usando valores padrão")
        return {
            'year': 0,
            'month': 0,
            'day': 0,
            'day_of_week': 0,
            'is_weekend': False
        }
        
    except Exception as e:
        logger.debug(f"Erro ao processar data '{date_str}': {e}")
        return {
            'year': 0,
            'month': 0,
            'day': 0,
            'day_of_week': 0,
            'is_weekend': False
        }

def safe_get_nested(data: Dict[str, Any], keys: List[str], default: Any = None) -> Any:
    """
    Acessa valores aninhados em um dicionário de forma segura
    
    Args:
        data: Dicionário de dados
        keys: Lista de chaves para navegar
        default: Valor padrão se a chave não existir
        
    Returns:
        Valor encontrado ou valor padrão
    """
    current = data
    for key in keys:
        if isinstance(current, dict) and key in current:
            current = current[key]
        else:
            return default
    return current

def create_summary_stats(data: pd.DataFrame, numeric_columns: List[str]) -> Dict[str, Any]:
    """
    Cria estatísticas resumidas para colunas numéricas
    
    Args:
        data: DataFrame com os dados
        numeric_columns: Lista de colunas numéricas
        
    Returns:
        Dicionário com estatísticas resumidas
    """
    summary = {}
    
    for col in numeric_columns:
        if col in data.columns:
            summary[col] = {
                'mean': float(data[col].mean()),
                'median': float(data[col].median()),
                'std': float(data[col].std()),
                'min': float(data[col].min()),
                'max': float(data[col].max()),
                'missing_count': int(data[col].isnull().sum()),
                'missing_percentage': float(data[col].isnull().sum() / len(data) * 100)
            }
    
    return summary

def validate_data_quality(data: pd.DataFrame) -> Dict[str, Any]:
    """
    Valida a qualidade dos dados
    
    Args:
        data: DataFrame para validação
        
    Returns:
        Dicionário com métricas de qualidade
    """
    total_rows = len(data)
    total_cols = len(data.columns)
    
    quality_report = {
        'total_rows': total_rows,
        'total_columns': total_cols,
        'missing_data': {},
        'duplicate_rows': int(data.duplicated().sum()),
        'duplicate_percentage': float(data.duplicated().sum() / total_rows * 100) if total_rows > 0 else 0.0
    }
    
    # Análise de dados ausentes por coluna
    for col in data.columns:
        missing_count = data[col].isnull().sum()
        missing_percentage = (missing_count / total_rows * 100) if total_rows > 0 else 0.0
        
        quality_report['missing_data'][col] = {
            'count': int(missing_count),
            'percentage': float(missing_percentage)
        }
    
    return quality_report

