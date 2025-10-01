# 🤖 Decision AI - Sistema de Recrutamento Inteligente

## 📋 Descrição do Projeto

O **Decision AI** é um sistema inteligente de recrutamento que utiliza técnicas avançadas de Machine Learning para automatizar e otimizar o processo de matching entre candidatos e vagas. O sistema analisa perfis de candidatos, requisitos de vagas e utiliza algoritmos de similaridade para identificar as melhores correspondências, reduzindo significativamente o tempo e esforço do processo de recrutamento.

### 🎯 Problema de Negócio

- **Processo manual ineficiente:** Recrutadores gastam horas analisando centenas de currículos
- **Matching impreciso:** Dificuldade em identificar candidatos com perfil ideal para cada vaga
- **Tempo de contratação elevado:** Processo demorado resulta em perda de talentos
- **Falta de padronização:** Critérios de avaliação inconsistentes entre recrutadores

### 💡 Solução Proposta

- **Matching automático:** Algoritmo de ML que analisa compatibilidade candidato-vaga
- **Análise inteligente:** Processamento de texto e extração de features relevantes
- **Interface intuitiva:** Dashboard interativo para visualização e gestão
- **Bot de entrevistas:** Sistema automatizado para triagem inicial
- **Relatórios analíticos:** Métricas e insights para otimização do processo

## 🛠️ Stack Tecnológica

### Backend & ML

- **Python 3.12+** - Linguagem principal
- **Pandas 2.0+** - Manipulação e análise de dados
- **NumPy 1.25+** - Computação numérica
- **Scikit-learn 1.3+** - Machine Learning
- **Joblib 1.3+** - Serialização de modelos

### Frontend & Interface

- **Streamlit 1.28+** - Interface web interativa
- **Plotly 5.15+** - Visualizações interativas
- **Matplotlib 3.7+** - Gráficos estáticos
- **Seaborn 0.12+** - Visualizações estatísticas

### Processamento de Texto

- **NLTK 3.8+** - Processamento de linguagem natural
- **TextBlob 0.17+** - Análise de sentimento
- **WordCloud 1.9+** - Visualização de palavras-chave

### Desenvolvimento & Qualidade

- **Jupyter 1.0+** - Notebooks de análise
- **Pytest 7.4+** - Testes automatizados
- **Black 23.0+** - Formatação de código
- **Flake8 6.0+** - Análise de qualidade

## 🚀 Como Rodar o App Localmente

### Pré-requisitos

- Python 3.12 ou superior
- Git (para clonar o repositório)

### 1. Clone o Repositório

```bash
git clone https://github.com/seu-usuario/decision-ai.git
cd decision-ai
```

### 2. Crie um Ambiente Virtual (Recomendado)

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/Mac
python -m venv venv
source venv/bin/activate
```

### 3. Instale as Dependências

```bash
pip install -r requirements.txt
```

### 4. Execute o Aplicativo

```bash
streamlit run app/app.py
```

### 5. Acesse o Sistema

Abra seu navegador e acesse: `http://localhost:8501`

## 📦 Instruções de Instalação Detalhadas

### Instalação Completa do Ambiente

1. **Instale o Python 3.12+**

   - Baixe em: https://www.python.org/downloads/
   - Marque "Add Python to PATH" durante a instalação

2. **Clone o Projeto**

   ```bash
   git clone https://github.com/seu-usuario/decision-ai.git
   cd decision-ai
   ```

3. **Configure o Ambiente Virtual**

   ```bash
   # Criar ambiente virtual
   python -m venv venv

   # Ativar ambiente virtual
   # Windows:
   venv\Scripts\activate
   # Linux/Mac:
   source venv/bin/activate
   ```

4. **Instale as Dependências**

   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

5. **Verifique a Instalação**
   ```bash
   python -c "import streamlit; print('Streamlit instalado com sucesso!')"
   ```

### Estrutura do Projeto

```
decision-ai/
├── app/                    # Aplicação Streamlit
│   ├── app.py             # Arquivo principal
│   └── models/            # Modelos treinados
├── src/                   # Código fonte
│   ├── preprocessing.py   # Pré-processamento
│   ├── feature_engineering.py  # Engenharia de features
│   ├── train.py          # Treinamento de modelos
│   ├── evaluate.py       # Avaliação de modelos
│   └── utils.py          # Utilitários
├── data/                  # Dados do projeto
│   ├── applicants.json   # Base de candidatos
│   ├── vagas.json        # Base de vagas
│   └── prospects.json    # Base de prospects
├── models/               # Modelos serializados
├── notebooks/            # Jupyter notebooks
├── tests/               # Testes automatizados
├── requirements.txt     # Dependências
└── README.md           # Este arquivo
```

## 🤖 Como Treinar o Modelo Novamente

### Método 1: Script Automatizado (Recomendado)

1. **Execute o Script de Treinamento**

   ```bash
   python train_simple_model.py
   ```

   Este script:
   - Carrega e processa automaticamente os dados
   - Treina múltiplos modelos (Random Forest, Gradient Boosting, Logistic Regression, SVM)
   - Seleciona o melhor modelo baseado no F1-Score
   - Salva o modelo serializado em `models/`
   - Cria um arquivo de status com informações do treinamento

2. **Verifique o Resultado**

   ```bash
   # Verifica se o modelo foi salvo
   ls models/
   
   # Verifica o status do treinamento
   cat models/training_status.txt
   ```

### Método 2: Via Interface Web

1. **Acesse o Sistema**

   - Execute: `streamlit run app/app.py`
   - Acesse: `http://localhost:8501`

2. **Navegue para Treinamento**

   - No menu lateral, clique em "🤖 Treinamento do Modelo"

3. **Configure os Parâmetros**

   - Escolha o algoritmo (Random Forest recomendado)
   - Defina o tamanho do conjunto de teste (padrão: 20%)
   - Configure o número de folds para validação cruzada (padrão: 5)

4. **Execute o Treinamento**

   - Clique em "🚀 Iniciar Treinamento"
   - Aguarde o processo (pode levar alguns minutos)
   - Visualize as métricas de performance

5. **Salve o Modelo**
   - O modelo será automaticamente salvo em `models/`
   - Nome do arquivo: `candidate_matcher_randomforest_YYYYMMDD_HHMMSS.joblib`

### Método 3: Via Código Python Personalizado

1. **Crie um Script de Treinamento**

   ```python
   # train_model.py
   from src.train import CandidateMatcherTrainer

   # Inicializa o treinador
   trainer = CandidateMatcherTrainer()

   # Executa pipeline completo
   result = trainer.run_full_training_pipeline(
       base_path=".",
       save_model=True
   )

   if result['success']:
       print(f"✅ Modelo treinado: {result['best_model_name']}")
       print(f"📊 Score F1: {result['best_score']:.4f}")
       print(f"📁 Salvo em: {result['model_path']}")
   else:
       print(f"❌ Erro: {result['error']}")
   ```

2. **Execute o Script**
   ```bash
   python train_model.py
   ```

### Método 4: Via Jupyter Notebook

1. **Abra o Notebook**

   ```bash
   jupyter notebook notebooks/model_training.ipynb
   ```

2. **Execute as Células**
   - Siga as instruções no notebook
   - Visualize os resultados interativamente
   - Ajuste parâmetros conforme necessário

### Parâmetros de Treinamento

#### Random Forest (Recomendado)

```python
{
    'n_estimators': 100,
    'max_depth': 10,
    'min_samples_split': 5,
    'min_samples_leaf': 2,
    'random_state': 42
}
```

#### Gradient Boosting

```python
{
    'n_estimators': 100,
    'learning_rate': 0.1,
    'max_depth': 6,
    'random_state': 42
}
```

#### Validação Cruzada

- **Folds:** 5 (padrão)
- **Métrica:** F1-Score
- **Estratificação:** Sim (mantém proporção de classes)

### Verificação do Modelo Treinado

1. **Verifique se o Modelo foi Salvo**

   ```bash
   ls models/
   # Deve mostrar arquivos .joblib
   ```

2. **Teste o Carregamento**

   ```python
   from src.model_utils import CandidateMatcher

   # Carrega o modelo
   matcher = CandidateMatcher()
   matcher.load_model('models/candidate_matcher_latest.joblib')

   # Verifica informações
   info = matcher.get_model_info()
   print(f"Modelo: {info['model_name']}")
   print(f"Score: {info['best_score']:.4f}")
   ```

## 📊 Funcionalidades do Sistema

### 🏠 Dashboard Principal

- Visão geral do sistema
- Métricas de performance
- Estatísticas de candidatos e vagas

### 🎯 Sistema de Matching Inteligente

- **Vaga → Candidatos:** Encontra candidatos ideais para uma vaga
- **Candidato → Vagas:** Identifica vagas adequadas para um candidato
- **Matching por Prospectos:** Análise da base de prospects

### 🤖 Bot de Entrevistas

- Sistema automatizado de triagem
- Perguntas inteligentes baseadas no perfil
- Análise de respostas em tempo real

### 📝 Análise de Entrevistas

- Processamento de transcrições
- Análise de sentimento
- Identificação de competências

### 📊 Análise Exploratória

- Visualizações interativas
- Estatísticas descritivas
- Insights de dados

### 🤖 Treinamento do Modelo

- Interface para retreinamento
- Ajuste de hiperparâmetros
- Validação cruzada

### 📈 Avaliação do Modelo

- Métricas de performance
- Gráficos de avaliação
- Análise de features importantes

## 🔧 Configurações Avançadas

### Variáveis de Ambiente

```bash
# .env
STREAMLIT_SERVER_PORT=8501
STREAMLIT_SERVER_ADDRESS=0.0.0.0
STREAMLIT_BROWSER_GATHER_USAGE_STATS=false
```

### Configurações de Performance

```python
# config.py
PERFORMANCE_CONFIG = {
    'batch_size': 1000,
    'n_jobs': -1,
    'memory_efficient': True
}
```

## 🧪 Testes

### Executar Testes

```bash
# Todos os testes
pytest

# Testes específicos
pytest tests/test_utils.py

# Com cobertura
pytest --cov=src tests/
```

### Testes de Integração

```bash
# Testa o pipeline completo
python -m pytest tests/test_integration.py -v
```

## 📈 Performance e Monitoramento

### Métricas do Sistema

- **Accuracy:** 85.2%
- **Precision:** 82.1%
- **Recall:** 88.3%
- **F1-Score:** 85.0%
- **ROC-AUC:** 0.91

### Monitoramento de Recursos

- **Memória:** ~2GB para dataset completo
- **CPU:** Otimizado para processamento paralelo
- **Tempo de resposta:** < 2s para matching

## 🚀 Deploy em Produção

### Docker (Recomendado)

```dockerfile
FROM python:3.12-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
EXPOSE 8501

CMD ["streamlit", "run", "app/app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

### Build e Execução

```bash
docker build -t decision-ai .
docker run -p 8501:8501 decision-ai
```

## 🤝 Contribuição

### Como Contribuir

1. Fork o projeto
2. Crie uma branch para sua feature (`git checkout -b feature/AmazingFeature`)
3. Commit suas mudanças (`git commit -m 'Add some AmazingFeature'`)
4. Push para a branch (`git push origin feature/AmazingFeature`)
5. Abra um Pull Request

### Padrões de Código

- Use Black para formatação: `black src/`
- Use Flake8 para linting: `flake8 src/`
- Escreva testes para novas funcionalidades
- Documente funções e classes

## 📄 Licença

Este projeto está sob a licença MIT. Veja o arquivo `LICENSE` para mais detalhes.

## 👥 Equipe

- **Desenvolvedor Principal:** [Seu Nome]
- **Orientador:** [Nome do Orientador]
- **Instituição:** FIAP - Faculdade de Informática e Administração Paulista

## 📞 Suporte

Para dúvidas ou problemas:

- **Email:** seu.email@exemplo.com
- **Issues:** [GitHub Issues](https://github.com/seu-usuario/decision-ai/issues)
- **Documentação:** [Wiki do Projeto](https://github.com/seu-usuario/decision-ai/wiki)

## 🔄 Changelog

### v2.0.0 (2024-09-27)

- ✅ Sistema de filtros inteligentes implementado
- ✅ Correção de erros de serialização PyArrow
- ✅ Interface otimizada para melhor performance
- ✅ Documentação completa atualizada

### v1.0.0 (2024-09-16)

- 🎉 Lançamento inicial
- ✅ Pipeline de ML completo
- ✅ Interface Streamlit funcional
- ✅ Sistema de matching básico

---

**Desenvolvido com ❤️ para otimizar o processo de recrutamento e conectar talentos às oportunidades ideais.**
