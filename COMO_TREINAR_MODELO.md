# 🚀 Como Treinar um Novo Modelo - Guia Rápido

## 📋 Pré-requisitos

Antes de treinar, verifique se você tem:

- ✅ Aplicação Streamlit rodando (`streamlit run app/app.py`)
- ✅ Dados carregados nos arquivos:
  - `applicants.json`
  - `vagas.json`
  - `prospects.json`
- ✅ Pacotes Python instalados (requirements.txt)

---

## 🎯 Passo a Passo: Treinar Novo Modelo

### 1️⃣ Acesse a Aplicação

Abra o navegador em: `http://localhost:8501`

---

### 2️⃣ Navegue até Treinamento

No menu lateral, selecione:

```
⚙️ Configurações → 🤖 Treinamento do Modelo de Matching
```

---

### 3️⃣ Clique em "Iniciar Treinamento"

Clique no botão vermelho:

```
🚀 Iniciar Treinamento do Modelo
```

---

### 4️⃣ Aguarde o Treinamento

O processo levará **1-2 minutos** e passará pelas seguintes etapas:

```
⏳ Preparando dados para treinamento...
⏳ Iniciando pré-processamento dos candidatos...
⏳ Pré-processamento concluído
⏳ Criando features de compatibilidade...
⏳ Treinando modelo: RandomForest
⏳ Treinando modelo: GradientBoosting
⏳ Treinando modelo: LogisticRegression
⏳ Treinando modelo: SVM
⏳ Melhor modelo selecionado: RandomForest
⏳ Salvando modelo...
✅ Modelo treinado com sucesso!
```

---

### 5️⃣ Verifique os Resultados

Após o treinamento, você verá:

#### Métricas do Modelo

```
Modelo Selecionado: RandomForest
F1-Score: 0.8534
```

#### Comparação de Modelos

Uma tabela comparando todos os modelos treinados:

| Modelo             | F1-Score | Accuracy | Precision | Recall |
| ------------------ | -------- | -------- | --------- | ------ |
| RandomForest       | 0.8534   | 0.8723   | 0.8456    | 0.8612 |
| GradientBoosting   | 0.8412   | 0.8645   | 0.8334    | 0.8491 |
| LogisticRegression | 0.7823   | 0.8034   | 0.7712    | 0.7935 |
| SVM                | 0.7645   | 0.7912   | 0.7534    | 0.7757 |

#### Gráfico de Performance

Um gráfico de barras comparando as métricas de todos os modelos.

---

### 6️⃣ Verifique o Modelo Salvo

Role para baixo até a seção **"📁 Modelo Existente"**:

```
✅ Modelo mais recente: candidate_matcher_randomforest_20251001_152334.joblib

Nome do Modelo: RandomForest
Score: 0.8534
Features: 125
Data de Treinamento: 2025-10-01T15:23:34
```

---

## 📁 Onde o Modelo é Salvo?

### Localização

```
GITHUB_UPLOAD_MINIMO/
├── models/
│   ├── candidate_matcher_latest.joblib ← Link para o mais recente
│   └── candidate_matcher_randomforest_20251001_152334.joblib ← Novo modelo
```

### Nome do Arquivo

O modelo é salvo com o formato:

```
candidate_matcher_{nome_do_modelo}_{data}_{hora}.joblib
```

Exemplo:

```
candidate_matcher_randomforest_20251001_152334.joblib
             ↑              ↑         ↑
         modelo          data      hora
```

---

## 🔄 Retreinar um Modelo

### Quando Retreinar?

Você deve retreinar quando:

- ✅ Novos candidatos foram adicionados
- ✅ Novas vagas foram criadas
- ✅ Dados de prospects atualizados
- ✅ Performance do modelo atual está baixa
- ✅ Mudanças nos critérios de matching

### Como Retreinar?

1. Simplesmente clique em **"🚀 Iniciar Treinamento do Modelo"** novamente
2. Um novo modelo será criado
3. O sistema usará automaticamente o modelo mais recente

---

## ⚙️ Configurações de Treinamento

### Modelos Testados

O sistema testa automaticamente 4 modelos:

1. **RandomForest** - Geralmente o melhor
2. **GradientBoosting** - Boa alternativa
3. **LogisticRegression** - Rápido e simples
4. **SVM** - Para casos específicos

### Seleção Automática

O sistema seleciona automaticamente o modelo com:

- ✅ Maior **F1-Score**
- ✅ Melhor **validação cruzada**

### Métricas Avaliadas

- **F1-Score**: Equilíbrio entre precisão e recall (principal métrica)
- **Accuracy**: Taxa geral de acertos
- **Precision**: Proporção de predições positivas corretas
- **Recall**: Proporção de casos positivos encontrados
- **ROC-AUC**: Área sob a curva ROC

---

## 🐛 Solução de Problemas

### ❌ Erro: "Nenhuma coluna válida restante"

**Solução:** O sistema usará dados sintéticos automaticamente.

```
⚠️ Nenhuma coluna válida restante após limpeza. Usando dataset sintético.
✅ Modelo treinado com sucesso!
```

### ❌ Erro: "Erro ao carregar modelo"

**Soluções:**

1. **Limpar Cache**

   - Clique no botão **"🔄 Limpar Cache"**
   - Aguarde a página recarregar

2. **Verificar Arquivos**

   ```powershell
   cd models
   dir *.joblib
   ```

3. **Retreinar**
   - Clique em **"🚀 Iniciar Treinamento do Modelo"** novamente

### ❌ Erro: "Dados não encontrados"

**Verificar:**

```powershell
# Verifique se os arquivos existem
dir applicants.json
dir vagas.json
dir prospects.json
```

**Solução:** Os arquivos JSON devem estar na raiz do projeto.

---

## 📊 Interpretando os Resultados

### F1-Score

| Valor       | Qualidade    | Ação                    |
| ----------- | ------------ | ----------------------- |
| > 0.90      | 🟢 Excelente | Modelo pronto!          |
| 0.80 - 0.90 | 🟢 Muito Bom | Modelo pronto!          |
| 0.70 - 0.80 | 🟡 Bom       | Pode melhorar           |
| 0.60 - 0.70 | 🟡 Regular   | Considere retreinar     |
| < 0.60      | 🔴 Baixo     | Retreine com mais dados |

### Validação Cruzada (CV)

A validação cruzada mostra:

- **CV Mean**: Score médio em 5 partes dos dados
- **CV Std**: Variação do score (quanto menor, mais estável)

**Exemplo:**

```
RandomForest - F1: 0.8534, CV F1: 0.8423 (+/- 0.0234)
                                    ↑         ↑
                                  média   variação
```

✅ **Ideal:** CV próximo do F1 e variação baixa

---

## 🎯 Dicas para Melhor Performance

### 1. Dados de Qualidade

- ✅ Mantenha dados atualizados
- ✅ Complete informações de candidatos
- ✅ Detalhe requisitos das vagas
- ✅ Atualize status dos prospects

### 2. Retreinamento Regular

- ✅ Retreine mensalmente
- ✅ Ou quando adicionar muitos dados novos
- ✅ Mantenha histórico de modelos

### 3. Monitoramento

- ✅ Acompanhe F1-Score ao longo do tempo
- ✅ Compare modelos antigos e novos
- ✅ Verifique se performance não cai

---

## 🔍 FAQ - Perguntas Frequentes

### P: Quanto tempo leva o treinamento?

**R:** Geralmente **1-2 minutos**, dependendo:

- Quantidade de dados
- Poder de processamento do computador
- Complexidade dos modelos

### P: Posso usar o sistema enquanto treina?

**R:** ✅ Sim! Abra outra aba do navegador em `localhost:8501`.

### P: Onde posso ver modelos antigos?

**R:** Na seção **"📁 Modelo Existente"**, clique em **"📋 Modelos disponíveis"**.

### P: O modelo antigo é apagado?

**R:** ❌ Não! Todos os modelos são mantidos na pasta `models/`.

### P: Qual modelo é usado pelo sistema?

**R:** Sempre o **mais recente** (por data de modificação).

### P: Posso forçar uso de modelo específico?

**R:** Sim! Renomeie o modelo desejado para `candidate_matcher_latest.joblib`.

```powershell
cd models
Copy-Item "candidate_matcher_randomforest_20250901_012534.joblib" "candidate_matcher_latest.joblib" -Force
```

---

## ✅ Checklist de Treinamento

Antes de treinar, verifique:

- [ ] Streamlit está rodando
- [ ] Arquivos JSON estão presentes (applicants, vagas, prospects)
- [ ] Dados estão atualizados
- [ ] Pasta `models/` existe
- [ ] Espaço em disco suficiente (> 100MB)

Durante o treinamento:

- [ ] Aguardar conclusão (não fechar navegador)
- [ ] Verificar se não há erros exibidos
- [ ] Acompanhar progresso das mensagens

Após o treinamento:

- [ ] Verificar F1-Score (idealmente > 0.80)
- [ ] Confirmar modelo salvo na seção "Modelo Existente"
- [ ] Testar no "Sistema de Matching"
- [ ] Verificar se resultados fazem sentido

---

## 🎉 Conclusão

Treinar um modelo é simples:

1. ✅ Clique em **"🚀 Iniciar Treinamento"**
2. ✅ Aguarde 1-2 minutos
3. ✅ Verifique os resultados
4. ✅ Use no Sistema de Matching!

**Pronto para treinar seu modelo? Vamos lá! 🚀**

---

**Última atualização:** 01/10/2025  
**Versão:** 1.0  
**Status:** ✅ Guia Completo
