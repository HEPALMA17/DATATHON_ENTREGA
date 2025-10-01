# 📁 Pasta de Modelos

## ⚠️ Importante: Modelos Não Incluídos

Os modelos treinados **NÃO** estão incluídos neste repositório porque são muito grandes (~206 MB cada) e excedem o limite do GitHub/Streamlit Cloud (25 MB).

## 🚀 Como Usar

### Opção 1: Treinar Novo Modelo (Recomendado)

Após fazer o deploy da aplicação:

1. **Acesse a aplicação** no Streamlit Cloud
2. **Navegue até:** `⚙️ Configurações` → `🤖 Treinamento do Modelo`
3. **Clique em:** `🚀 Iniciar Treinamento do Modelo`
4. **Aguarde:** 1-2 minutos
5. **Pronto!** O modelo será criado e salvo automaticamente

### Opção 2: Desenvolvimento Local

Se você está desenvolvendo localmente, treine um modelo:

```bash
# Inicie a aplicação
streamlit run app/app.py

# Acesse: http://localhost:8501
# Navegue até: Treinamento do Modelo
# Clique em: Iniciar Treinamento
```

## 📊 O Que Acontece Sem Modelo?

A aplicação **funciona normalmente** mesmo sem modelo treinado:

- ✅ **Dashboard** → Funciona com visualizações dos dados
- ✅ **Análise de Candidatos** → Funciona com dados reais
- ✅ **Análise de Vagas** → Funciona com estatísticas
- ⚠️ **Sistema de Matching** → Usa matching determinístico baseado em hash
  - Não usa IA/ML
  - Scores são gerados deterministicamente
  - Ainda funciona, mas sem o poder preditivo do modelo

## 🎯 Após Treinar

O modelo será salvo nesta pasta com o formato:

```
models/
├── candidate_matcher_randomforest_YYYYMMDD_HHMMSS.joblib
└── candidate_matcher_latest.joblib
```

## 🔒 Segurança

Esta pasta contém apenas modelos de ML. Nenhum dado sensível é armazenado aqui.

- ✅ Modelos podem ser compartilhados
- ✅ Modelos podem ser versionados (se pequenos)
- ❌ Modelos grandes (>25MB) não devem ir para GitHub

## 📝 Configuração do .gitignore

Os modelos `.joblib` já estão configurados para serem ignorados pelo Git:

```gitignore
# Modelos ML (muito grandes para GitHub)
models/*.joblib
!models/README.md
```

## 🆘 Problemas?

### Modelo não carrega após treinar

1. Clique em **🔄 Limpar Cache**
2. Recarregue a página
3. Verifique a seção "📁 Modelo Existente"

### Erro durante treinamento

O sistema usa **dados sintéticos** como fallback se houver problemas com os dados reais. O treinamento sempre completará com sucesso.

### Quer modelo pré-treinado

Se você tem um modelo treinado localmente e quer usá-lo:

1. Copie o arquivo `.joblib` para esta pasta
2. Renomeie para `candidate_matcher_latest.joblib`
3. Faça upload manual se necessário

---

**Última atualização:** 01/10/2025
