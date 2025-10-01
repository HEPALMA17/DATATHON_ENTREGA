"""
Arquivo de configuração para o projeto Decision AI
"""

import os
from pathlib import Path

# Diretórios do projeto
BASE_DIR = Path(__file__).parent
SRC_DIR = BASE_DIR / "src"
APP_DIR = BASE_DIR / "app"
MODELS_DIR = BASE_DIR / "models"
NOTEBOOKS_DIR = BASE_DIR / "notebooks"
TESTS_DIR = BASE_DIR / "tests"

# Configurações de dados
DATA_FILES = {
    'applicants': 'applicants.json',
    'vagas': 'vagas.json',
    'prospects': 'prospects.json'
}

# Configurações de ML
ML_CONFIG = {
    'test_size': 0.2,
    'random_state': 42,
    'cv_folds': 5,
    'scoring': 'f1'
}

# Configurações de features
FEATURE_CONFIG = {
    'max_text_features': 100,
    'pca_components': 50,
    'min_df': 2,
    'ngram_range': (1, 2)
}

# Skills técnicas para extração
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

# Configurações de logging
LOGGING_CONFIG = {
    'level': 'INFO',
    'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    'file': 'decision_ai.log'
}

# Configurações do Streamlit
STREAMLIT_CONFIG = {
    'page_title': 'Decision AI - Sistema de Recrutamento Inteligente',
    'page_icon': '🤖',
    'layout': 'wide',
    'initial_sidebar_state': 'expanded'
}

# Configurações de avaliação
EVALUATION_CONFIG = {
    'metrics': ['accuracy', 'precision', 'recall', 'f1', 'roc_auc'],
    'plots': ['confusion_matrix', 'roc_curve', 'precision_recall', 'feature_importance'],
    'output_dir': 'evaluation_reports'
}

# Configurações de modelo
MODEL_CONFIG = {
    'save_format': 'joblib',
    'latest_model_name': 'candidate_matcher_latest.joblib',
    'backup_models': True,
    'max_backup_models': 5
}

# Configurações de performance
PERFORMANCE_CONFIG = {
    'batch_size': 1000,
    'n_jobs': -1,
    'memory_efficient': True
}

# Configurações de segurança
SECURITY_CONFIG = {
    'validate_input': True,
    'sanitize_output': True,
    'log_sensitive_operations': False
}

# Configurações de deploy
DEPLOY_CONFIG = {
    'production': False,
    'debug': True,
    'host': '0.0.0.0',
    'port': 8501
}

def get_config():
    """Retorna configuração completa do projeto"""
    return {
        'base_dir': str(BASE_DIR),
        'src_dir': str(SRC_DIR),
        'app_dir': str(APP_DIR),
        'models_dir': str(MODELS_DIR),
        'notebooks_dir': str(NOTEBOOKS_DIR),
        'tests_dir': str(TESTS_DIR),
        'data_files': DATA_FILES,
        'ml_config': ML_CONFIG,
        'feature_config': FEATURE_CONFIG,
        'tech_skills': TECH_SKILLS,
        'logging_config': LOGGING_CONFIG,
        'streamlit_config': STREAMLIT_CONFIG,
        'evaluation_config': EVALUATION_CONFIG,
        'model_config': MODEL_CONFIG,
        'performance_config': PERFORMANCE_CONFIG,
        'security_config': SECURITY_CONFIG,
        'deploy_config': DEPLOY_CONFIG
    }

def ensure_directories():
    """Cria diretórios necessários se não existirem"""
    directories = [SRC_DIR, APP_DIR, MODELS_DIR, NOTEBOOKS_DIR, TESTS_DIR]
    
    for directory in directories:
        directory.mkdir(exist_ok=True)
        print(f"✅ Diretório verificado/criado: {directory}")

if __name__ == "__main__":
    # Testa configuração
    config = get_config()
    print("🔧 Configuração do Decision AI:")
    print(f"   • Diretório base: {config['base_dir']}")
    print(f"   • Skills técnicas: {len(config['tech_skills'])}")
    print(f"   • Configurações ML: {len(config['ml_config'])}")
    
    # Cria diretórios
    ensure_directories()
    
    print("✅ Configuração carregada com sucesso!")




