"""
Testes unitários para o módulo utils.py
"""

import unittest
import pandas as pd
import numpy as np
from datetime import datetime
import sys
import os

# Adiciona o diretório src ao path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from utils import (
    clean_text, 
    extract_skills_from_text, 
    calculate_text_similarity,
    normalize_score,
    create_date_features,
    safe_get_nested,
    create_summary_stats,
    validate_data_quality
)

class TestUtils(unittest.TestCase):
    """Testes para as funções utilitárias"""
    
    def test_clean_text(self):
        """Testa a função clean_text"""
        # Teste com texto normal
        self.assertEqual(clean_text("Hello World!"), "hello world")
        
        # Teste com texto com caracteres especiais
        self.assertEqual(clean_text("Python & Machine Learning"), "python machine learning")
        
        # Teste com múltiplos espaços
        self.assertEqual(clean_text("  Python    ML  "), "python ml")
        
        # Teste com string vazia
        self.assertEqual(clean_text(""), "")
        
        # Teste com None
        self.assertEqual(clean_text(None), "")
        
        # Teste com número
        self.assertEqual(clean_text(123), "")
    
    def test_extract_skills_from_text(self):
        """Testa a função extract_skills_from_text"""
        skills_keywords = ['python', 'java', 'machine learning', 'sql']
        
        # Teste com skills encontradas
        text = "Experiência em Python e Machine Learning"
        found_skills = extract_skills_from_text(text, skills_keywords)
        self.assertEqual(set(found_skills), {'python', 'machine learning'})
        
        # Teste sem skills
        text = "Experiência em marketing e vendas"
        found_skills = extract_skills_from_text(text, skills_keywords)
        self.assertEqual(found_skills, [])
        
        # Teste com texto vazio
        found_skills = extract_skills_from_text("", skills_keywords)
        self.assertEqual(found_skills, [])
        
        # Teste com None
        found_skills = extract_skills_from_text(None, skills_keywords)
        self.assertEqual(found_skills, [])
    
    def test_calculate_text_similarity(self):
        """Testa a função calculate_text_similarity"""
        # Teste com textos similares
        similarity = calculate_text_similarity("python machine learning", "python ml")
        self.assertGreater(similarity, 0.0)
        self.assertLessEqual(similarity, 1.0)
        
        # Teste com textos idênticos
        similarity = calculate_text_similarity("python", "python")
        self.assertEqual(similarity, 1.0)
        
        # Teste com textos completamente diferentes
        similarity = calculate_text_similarity("python", "marketing")
        self.assertEqual(similarity, 0.0)
        
        # Teste com texto vazio
        similarity = calculate_text_similarity("", "python")
        self.assertEqual(similarity, 0.0)
        
        # Teste com ambos vazios
        similarity = calculate_text_similarity("", "")
        self.assertEqual(similarity, 0.0)
    
    def test_normalize_score(self):
        """Testa a função normalize_score"""
        # Teste normal
        normalized = normalize_score(75, 0, 100)
        self.assertEqual(normalized, 0.75)
        
        # Teste com valor mínimo
        normalized = normalize_score(0, 0, 100)
        self.assertEqual(normalized, 0.0)
        
        # Teste com valor máximo
        normalized = normalize_score(100, 0, 100)
        self.assertEqual(normalized, 1.0)
        
        # Teste com valores iguais
        normalized = normalize_score(50, 50, 50)
        self.assertEqual(normalized, 0.0)
        
        # Teste com valor fora do range
        normalized = normalize_score(150, 0, 100)
        self.assertEqual(normalized, 1.0)
        
        # Teste com valor negativo
        normalized = normalize_score(-10, 0, 100)
        self.assertEqual(normalized, 0.0)
    
    def test_create_date_features(self):
        """Testa a função create_date_features"""
        # Teste com data válida
        date_str = "15-03-2023"
        features = create_date_features(date_str)
        
        self.assertEqual(features['year'], 2023)
        self.assertEqual(features['month'], 3)
        self.assertEqual(features['day'], 15)
        self.assertIsInstance(features['day_of_week'], int)
        self.assertIsInstance(features['is_weekend'], bool)
        
        # Teste com string vazia
        features = create_date_features("")
        self.assertEqual(features['year'], 0)
        self.assertEqual(features['month'], 0)
        self.assertEqual(features['day'], 0)
        
        # Teste com None
        features = create_date_features(None)
        self.assertEqual(features['year'], 0)
        self.assertEqual(features['month'], 0)
        self.assertEqual(features['day'], 0)
        
        # Teste com data inválida
        features = create_date_features("data-invalida")
        self.assertEqual(features['year'], 0)
        self.assertEqual(features['month'], 0)
        self.assertEqual(features['day'], 0)
    
    def test_safe_get_nested(self):
        """Testa a função safe_get_nested"""
        data = {
            'level1': {
                'level2': {
                    'level3': 'value'
                }
            }
        }
        
        # Teste com caminho válido
        value = safe_get_nested(data, ['level1', 'level2', 'level3'])
        self.assertEqual(value, 'value')
        
        # Teste com caminho parcial
        value = safe_get_nested(data, ['level1', 'level2'])
        self.assertEqual(value, {'level3': 'value'})
        
        # Teste com caminho inválido
        value = safe_get_nested(data, ['level1', 'invalid'])
        self.assertIsNone(value)
        
        # Teste com caminho vazio
        value = safe_get_nested(data, [])
        self.assertEqual(value, data)
        
        # Teste com valor padrão
        value = safe_get_nested(data, ['level1', 'invalid'], 'default')
        self.assertEqual(value, 'default')
    
    def test_create_summary_stats(self):
        """Testa a função create_summary_stats"""
        # Cria DataFrame de teste
        df = pd.DataFrame({
            'numeric_col': [1, 2, 3, 4, 5],
            'text_col': ['a', 'b', 'c', 'd', 'e']
        })
        
        # Teste com colunas numéricas
        stats = create_summary_stats(df, ['numeric_col'])
        
        self.assertIn('numeric_col', stats)
        self.assertEqual(stats['numeric_col']['mean'], 3.0)
        self.assertEqual(stats['numeric_col']['min'], 1.0)
        self.assertEqual(stats['numeric_col']['max'], 5.0)
        self.assertEqual(stats['numeric_col']['missing_count'], 0)
        
        # Teste com coluna inexistente
        stats = create_summary_stats(df, ['inexistent_col'])
        self.assertEqual(stats, {})
    
    def test_validate_data_quality(self):
        """Testa a função validate_data_quality"""
        # Cria DataFrame de teste com dados ausentes
        df = pd.DataFrame({
            'col1': [1, 2, None, 4, 5],
            'col2': ['a', 'b', 'c', None, 'e'],
            'col3': [1, 1, 1, 1, 1]  # Valores duplicados
        })
        
        quality_report = validate_data_quality(df)
        
        self.assertEqual(quality_report['total_rows'], 5)
        self.assertEqual(quality_report['total_columns'], 3)
        self.assertEqual(quality_report['duplicate_rows'], 0)
        
        # Verifica dados ausentes
        self.assertIn('col1', quality_report['missing_data'])
        self.assertEqual(quality_report['missing_data']['col1']['count'], 1)
        self.assertEqual(quality_report['missing_data']['col1']['percentage'], 20.0)
        
        self.assertIn('col2', quality_report['missing_data'])
        self.assertEqual(quality_report['missing_data']['col2']['count'], 1)
        self.assertEqual(quality_report['missing_data']['col2']['percentage'], 20.0)
        
        self.assertIn('col3', quality_report['missing_data'])
        self.assertEqual(quality_report['missing_data']['col3']['count'], 0)
        self.assertEqual(quality_report['missing_data']['col3']['percentage'], 0.0)

if __name__ == '__main__':
    unittest.main()




