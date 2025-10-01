# 🚀 Deploy no Streamlit Cloud - Guia Completo

## ✅ Pré-requisitos Verificados

Antes de fazer o deploy, confirme:

### Tamanhos de Arquivo (Limite: 25 MB)

- ✅ `applicants.json` → 2.49 MB ✓
- ✅ `prospects.json` → 0.62 MB ✓
- ✅ `vagas.json` → 0.74 MB ✓
- ✅ `decision_ai_model.joblib` → 0.05 MB ✓
- ❌ Modelos grandes removidos (eram 206 MB cada)

### Arquivos de Configuração

- ✅ `requirements.txt` → Pronto
- ✅ `packages.txt` → Pronto (se necessário)
- ✅ `.gitignore` → Configurado
- ✅ `README.md` → Completo

---

## 📋 Passo a Passo: Deploy

### 1️⃣ Preparar Repositório GitHub

#### A. Criar Repositório

```bash
# Vá para: https://github.com/new
# Nome: datathon-decision-ai (ou outro nome)
# Visibilidade: Public
# ✅ Criar repositório
```

#### B. Inicializar Git Local

```bash
cd "C:\Users\Samsung\Desktop\#TREINAMENTO\POS FIAP\DATATHON - ENTREGA\DEPLOY_DATATHON\GITHUB_UPLOAD_MINIMO"

git init
git add .
git commit -m "Initial commit - Decision AI Sistema de Recrutamento"
```

#### C. Conectar com GitHub

```bash
# Substitua SEU_USUARIO pelo seu username do GitHub
git remote add origin https://github.com/SEU_USUARIO/datathon-decision-ai.git
git branch -M main
git push -u origin main
```

---

### 2️⃣ Deploy no Streamlit Cloud

#### A. Acessar Streamlit Cloud

1. Vá para: https://share.streamlit.io/
2. Faça login com GitHub
3. Clique em **"New app"**

#### B. Configurar Deploy

```
Repository: SEU_USUARIO/datathon-decision-ai
Branch: main
Main file path: app/app.py
```

#### C. Advanced Settings (Opcional)

```
Python version: 3.10
```

#### D. Deploy!

Clique em **"Deploy!"**

⏳ **Aguarde:** 2-5 minutos para o deploy completar

---

### 3️⃣ Após o Deploy

#### A. Treinar Modelo

A aplicação vai subir **SEM** modelo treinado (arquivos muito grandes). Você precisa treinar:

1. **Acesse a aplicação** no link fornecido pelo Streamlit
2. **Navegue até:** `⚙️ Configurações` → `🤖 Treinamento do Modelo`
3. **Clique em:** `🚀 Iniciar Treinamento do Modelo`
4. **Aguarde:** 1-2 minutos
5. **Modelo criado!** ✅

#### B. Verificar Funcionalidades

Teste todas as páginas:

- ✅ Dashboard Principal
- ✅ Análise de Candidatos
- ✅ Análise de Vagas
- ✅ Sistema de Matching (após treinar modelo)
- ✅ Treinamento do Modelo

---

## 🔧 Configurações Importantes

### `requirements.txt`

```txt
streamlit
pandas
numpy
scikit-learn
joblib
plotly
matplotlib
seaborn
nltk
textblob
```

### `packages.txt` (se necessário)

```txt
build-essential
```

### `.streamlit/config.toml`

```toml
[server]
headless = true
port = 8501
enableCORS = false
enableXsrfProtection = false

[browser]
gatherUsageStats = false
```

---

## 📊 Tamanhos de Arquivo - Checklist

| Arquivo                         | Tamanho    | Status      | Limite  |
| ------------------------------- | ---------- | ----------- | ------- |
| applicants.json                 | 2.49 MB    | ✅ OK       | < 25 MB |
| prospects.json                  | 0.62 MB    | ✅ OK       | < 25 MB |
| vagas.json                      | 0.74 MB    | ✅ OK       | < 25 MB |
| decision_ai_model.joblib        | 0.05 MB    | ✅ OK       | < 25 MB |
| ~~candidate*matcher*\*.joblib~~ | ~~206 MB~~ | ❌ Removido | > 25 MB |

**Total do repositório:** ~4 MB ✅

---

## ⚠️ Problemas Comuns e Soluções

### Erro: "File too large"

**Causa:** Algum arquivo excede 25 MB

**Solução:**

```bash
# Verifique tamanhos
Get-ChildItem -Recurse | Where-Object {$_.Length -gt 25MB} | Select-Object FullName, @{Name="SizeMB";Expression={[math]::Round($_.Length/1MB, 2)}}

# Adicione ao .gitignore
echo "caminho/arquivo_grande.ext" >> .gitignore
```

### Erro: "Module not found"

**Causa:** Falta pacote no `requirements.txt`

**Solução:**

1. Edite `requirements.txt`
2. Adicione o pacote faltando
3. Commit e push
4. Streamlit fará redeploy automático

### Erro: "Application error"

**Causa:** Erro no código

**Solução:**

1. Verifique logs no Streamlit Cloud
2. Teste localmente: `streamlit run app/app.py`
3. Corrija e faça push

### Modelo não carrega

**Solução:**

1. Modelos grandes não estão no repo
2. Treine um novo modelo na aplicação deployada
3. Use a seção "Treinamento do Modelo"

---

## 🎯 Após Deploy Bem-Sucedido

### Compartilhar Aplicação

```
URL pública: https://seu-app.streamlit.app
```

### Domínio Customizado (Opcional)

No Streamlit Cloud:

1. Settings → General
2. Custom subdomain: `decision-ai`
3. Resultado: `https://decision-ai.streamlit.app`

### Análise de Uso

No Streamlit Cloud:

- Veja estatísticas de uso
- Monitore performance
- Verifique logs

---

## 🔄 Atualizações Futuras

### Fazer Update

```bash
# Faça mudanças no código
git add .
git commit -m "Descrição da mudança"
git push

# Streamlit fará redeploy automático
```

### Rollback se Necessário

```bash
git revert HEAD
git push
```

---

## 📈 Otimizações para Produção

### 1. Cache Agressivo

```python
@st.cache_data(ttl=3600)  # Cache por 1 hora
def load_data():
    ...

@st.cache_resource
def load_model():
    ...
```

### 2. Lazy Loading

Carregue dados apenas quando necessário

### 3. Compressão

Considere comprimir arquivos JSON grandes

### 4. CDN

Use CDN para assets estáticos se houver

---

## 🆘 Suporte

### Documentação Oficial

- Streamlit: https://docs.streamlit.io/
- Deploy: https://docs.streamlit.io/streamlit-community-cloud/get-started

### Comunidade

- Forum: https://discuss.streamlit.io/
- GitHub Issues: https://github.com/streamlit/streamlit/issues

### Logs

Acesse logs no Streamlit Cloud:

```
Your app → Menu (⋮) → Manage app → Logs
```

---

## ✅ Checklist Final de Deploy

Antes de fazer push final:

- [ ] Todos os arquivos < 25 MB
- [ ] `requirements.txt` completo
- [ ] `.gitignore` configurado
- [ ] Modelos grandes removidos/ignorados
- [ ] README.md atualizado
- [ ] Código testado localmente
- [ ] Sem credenciais hardcoded
- [ ] Logs não verbosos demais

Durante o deploy:

- [ ] Repository URL correto
- [ ] Branch correto (main)
- [ ] Main file path: `app/app.py`
- [ ] Deploy iniciou sem erros

Após o deploy:

- [ ] Aplicação carregou
- [ ] Todas as páginas funcionam
- [ ] Treinou modelo
- [ ] Sistema de matching funciona
- [ ] Sem erros nos logs

---

## 🎉 Pronto!

Sua aplicação Decision AI está no ar! 🚀

**URL:** `https://seu-app.streamlit.app`

---

**Última atualização:** 01/10/2025  
**Versão:** 1.0  
**Status:** ✅ Pronto para Deploy
