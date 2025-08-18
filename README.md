# 🚀 ML Pipeline Orchestrator

Uma aplicação web interativa para configurar, executar e avaliar pipelines de Machine Learning com interface Streamlit.

## ✨ Características Principais

### 🆕 Novas Funcionalidades
- **💾 Salvamento de Modelos**: Salve modelos treinados com todas as informações (scaler, feature selector, métricas)
- **🔄 Reutilização de Modelos**: Carregue modelos salvos e use para novas predições
- **🔮 Sistema de Predições**: Interface para predições individuais ou em lote
- **📊 Métricas em Destaque**: Visualização clara da performance dos modelos
- **🗂️ Gerenciamento de Modelos**: Liste, compare e gerencie todos os modelos salvos

### 🔧 Funcionalidades Principais
- **🔧 Configuração Visual**: Interface intuitiva para configurar todos os aspectos do pipeline
- **📊 Múltiplos Algoritmos**: Random Forest, Logistic Regression, SVM
- **📈 Visualizações Interativas**: Gráficos de avaliação com Plotly
- **⚙️ Configuração YAML**: Pipelines reproduzíveis via arquivos de configuração
- **📝 Logging Detalhado**: Rastreamento completo de todas as etapas
- **🎯 Métricas Abrangentes**: Accuracy, Precision, Recall, F1, ROC-AUC
- **📂 Upload de Dados**: Suporte a datasets personalizados
- **🔄 Cross-Validation**: Validação cruzada integrada

## 🏗️ Estrutura do Projeto

```
ml_pipeline_project/
├── src/
│   ├── app.py                    # Interface Streamlit principal
│   ├── pipeline/
│   │   ├── orchestrator.py       # Orquestrador do pipeline
│   │   ├── data_processor.py     # Processamento de dados
│   │   ├── model_trainer.py      # Treinamento de modelos
│   │   └── evaluator.py         # Avaliação de modelos
│   └── utils/
│       ├── logger.py            # Sistema de logging
│       ├── config_loader.py     # Carregamento de configurações
│       └── model_manager.py     # Gerenciamento de modelos
├── configs/
│   └── default_config.yaml      # Configuração padrão
├── data/
│   ├── raw/                     # Dados brutos
│   ├── processed/               # Dados processados
│   ├── results/                 # Resultados e relatórios
│   └── models/                  # Modelos salvos
├── examples/
│   └── example_usage.py         # Exemplos de uso
├── logs/                        # Arquivos de log
└── requirements.txt
```

## 🚀 Instalação e Execução

### 1. Clone e Setup

```bash
# Clonar repositório (ou criar diretório)
mkdir ml_pipeline_project
cd ml_pipeline_project

# Executar setup
python setup.py
```

### 2. Instalar Dependências

```bash
# Criar ambiente virtual
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows

# Instalar dependências
pip install -r requirements.txt
```

### 3. Executar Aplicação

```bash
# Navegar para src
cd src

# Executar Streamlit
streamlit run app.py
```

**Acesse**: http://localhost:8501

## 🎯 Como Usar

### 🆕 Modos de Operação

A aplicação agora possui 3 modos principais:

#### 1. **🎓 Treinar Novo Modelo**
- Escolha dataset (Iris, Wine, Breast Cancer ou upload personalizado)
- Configure algoritmo e parâmetros
- Define métricas de avaliação e cross-validation
- **Novo**: Opção de salvar o modelo automaticamente
- Visualize resultados e métricas em destaque

#### 2. **📁 Usar Modelo Salvo**  
- Liste todos os modelos salvos com suas informações
- Carregue modelo específico com um clique
- Veja detalhes: algoritmo, acurácia, features, data de criação
- Modelo fica pronto para fazer predições

#### 3. **🔮 Fazer Predições**
- **Entrada Manual**: Insira valores das features individualmente
- **Upload CSV**: Faça predições em lote com arquivo CSV
- Visualize probabilidades por classe
- Download dos resultados em CSV
- Gráficos de distribuição das predições

### 📊 Visualização de Resultados

#### Métricas em Destaque
- **Performance Cards**: Accuracy, F1, Precision, Recall com cores indicativas
- **Gauge ROC-AUC**: Medidor visual da área sob a curva ROC
- **Recomendações Automáticas**: Sugestões baseadas na performance

#### Visualizações Interativas
- Matriz de confusão com valores e percentuais
- Curvas ROC multiclasse
- Importância das features
- Curvas Precision-Recall
- Distribuição de probabilidades

### 💾 Gerenciamento de Modelos

#### Salvamento Inteligente
- Modelo principal (.pkl)
- Scaler de preprocessamento
- Feature selector
- Informações completas (JSON)
- Métricas de performance
- Configuração utilizada

#### Carregamento e Reutilização
- Lista modelos com preview das informações
- Carregamento automático de preprocessadores
- Validação de features necessárias
- Aplicação automática de transformações

## 📊 Datasets Incluídos

### Clássicos (Sklearn)
- **Iris**: 150 amostras, 4 features, 3 classes - Perfeito para iniciantes
- **Wine**: 178 amostras, 13 features, 3 classes - Classificação química
- **Breast Cancer**: 569 amostras, 30 features, 2 classes - Diagnóstico médico

### Personalizados (Projetos Reais)
- **Credit Scoring**: Análise de risco de crédito
- **Hypertension**: Predição de hipertensão arterial  
- **Phone Addiction**: Detecção de vício em smartphones

### Upload Personalizado
- Suporte a qualquer CSV
- Pré-processamento automático
- Detecção inteligente de target
- Conversão de tipos de dados

## ⚙️ Configuração via YAML

### Configuração Completa de Exemplo

```yaml
project:
  name: "Pipeline Avançado"
  version: "2.0.0"

data:
  source: "wine"
  test_size: 0.3
  random_state: 42

preprocessing:
  scaling:
    method: "standard"  # standard, minmax, robust
  feature_selection:
    enabled: true
    method: "selectkbest"  # selectkbest, rfe
    k: 8
  handle_missing:
    strategy: "mean"  # mean, median, drop

model:
  algorithm: "random_forest"
  random_forest:
    n_estimators: 200
    max_depth: 15
    min_samples_split: 5
    random_state: 42

training:
  cross_validation:
    enabled: true
    cv_folds: 10
  hyperparameter_tuning:
    enabled: true
    method: "grid_search"  # grid_search, random_search

evaluation:
  metrics:
    - "accuracy"
    - "precision"
    - "recall" 
    - "f1"
    - "roc_auc"
  plots:
    - "confusion_matrix"
    - "roc_curve"
    - "feature_importance"
    - "precision_recall_curve"

pipeline_steps:
  - "load_data"
  - "preprocess_data"
  - "train_model" 
  - "evaluate_model"
  - "save_results"

output:
  save_model: true
  results_dir: "data/models"
```

## 🤖 Algoritmos Suportados

### **Random Forest**
```yaml
random_forest:
  n_estimators: 100     # Número de árvores
  max_depth: 10         # Profundidade máxima
  min_samples_split: 2  # Amostras mín. para split
  min_samples_leaf: 1   # Amostras mín. por folha
  random_state: 42
```

### **Logistic Regression**
```yaml
logistic_regression:
  C: 1.0               # Regularização
  solver: "lbfgs"      # Algoritmo de otimização
  max_iter: 1000       # Iterações máximas
  random_state: 42
```

### **Support Vector Machine**
```yaml
svm:
  C: 1.0               # Parâmetro de regularização
  kernel: "rbf"        # rbf, linear, poly
  gamma: "scale"       # Coeficiente do kernel
  probability: true    # Para predições probabilísticas
  random_state: 42
```

## 🔮 Sistema de Predições

### Predição Individual
```python
# Via interface web:
# 1. Selecione "Fazer Predições"
# 2. Escolha "Entrada Manual"
# 3. Insira valores das features
# 4. Clique "Fazer Predição"

# Resultado:
# - Classe predita
# - Probabilidades por classe
# - Gráfico de barras das probabilidades
```

### Predições em Lote
```python
# Via interface web:
# 1. Prepare CSV com as mesmas features do modelo
# 2. Selecione "Upload CSV"
# 3. Faça upload do arquivo
# 4. Clique "Fazer Predições"

# Resultado:
# - Tabela com todas as predições
# - Probabilidades por classe
# - Gráfico de distribuição
# - Download dos resultados
```

### Via Código Python
```python
from src.utils.model_manager import ModelManager
import pandas as pd

# Inicializar gerenciador
manager = ModelManager()

# Carregar dados
data = pd.read_csv("novos_dados.csv")

# Fazer predições
results = manager.predict("meu_modelo", data)

print(f"Predições: {results['predictions']}")
print(f"Probabilidades: {results['probabilities']}")
```

## 💡 Exemplos Práticos

### Executar Exemplos Completos
```bash
python examples/example_usage.py
```

Este script demonstra:
1. ✅ Treinamento e salvamento de modelos
2. ✅ Carregamento e predições
3. ✅ Comparação de algoritmos
4. ✅ Predições em lote
5. ✅ Gerenciamento de modelos

### Workflow Típico

#### 1. Exploração de Dados
```python
# Carregue dataset na interface
# Visualize estatísticas e gráficos
# Analise qualidade dos dados
# Identifique padrões e correlações
```

#### 2. Experimentação de Modelos
```python
# Teste diferentes algoritmos
# Compare performance
# Ajuste hiperparâmetros
# Use cross-validation
```

#### 3. Modelo Final
```python
# Selecione melhor algoritmo
# Salve modelo com nome descritivo
# Exporte configuração
# Documente resultados
```

#### 4. Produção
```python
# Carregue modelo salvo
# Faça predições em novos dados
# Monitore performance
# Atualize quando necessário
```

## 🎨 Interface Web - Guia Visual

### Sidebar - Modos de Operação
```
⚙️ Configurações do Pipeline
├── 🎯 Modo: [Treinar|Carregar|Predizer]
├── 📂 Dados
│   ├── Fonte: [iris|wine|upload...]
│   └── 📤 Upload CSV
├── 🤖 Modelo  
│   ├── Algoritmo: [RF|LogReg|SVM]
│   └── Parâmetros específicos
├── 📊 Avaliação
│   ├── Test Size: [0.1 - 0.5]
│   ├── CV Folds: [3 - 10] 
│   └── Métricas: [☑️ accuracy ☑️ f1...]
└── 🚀 Executar Pipeline
```

### Main Area - Resultados
```
📊 Resultados do Pipeline
├── 📈 Cards de Métricas
│   ├── ✅ Accuracy: 0.9567 (Excelente)
│   ├── 📊 F1-Score: 0.9234 (Bom)  
│   ├── 🎯 Precision: 0.9445
│   └── 📈 Recall: 0.9123
├── 🌟 ROC-AUC Gauge: 0.94
├── 💡 Recomendações
└── 📑 Tabs: [Métricas|Visualizações|Config|Logs]
```

## 📊 Métricas e Interpretação

### Performance Cards
- **🟢 Excelente**: ≥ 0.90 (Verde)
- **🟡 Bom**: 0.80-0.89 (Amarelo)  
- **🟠 Regular**: 0.70-0.79 (Laranja)
- **🔴 Precisa Melhorar**: < 0.70 (Vermelho)

### ROC-AUC Gauge
- **0.9-1.0**: Excelente discriminação
- **0.8-0.9**: Boa discriminação
- **0.7-0.8**: Discriminação razoável
- **0.6-0.7**: Discriminação pobre
- **≤0.5**: Sem discriminação

### Recomendações Automáticas
```
✅ Excelente acurácia! Modelo performando muito bem.
⚠️ Recall baixo. Considere balanceamento de classes.
📊 Modelo conservador - boa precisão, recall menor.
⚠️ Alta variabilidade no CV. Possível overfitting.
```

## 🔧 Extensões e Personalização

### Adicionando Novos Algoritmos
```python
# src/pipeline/model_trainer.py
def _initialize_model(self, algorithm):
    # ... algoritmos existentes ...
    elif algorithm == 'xgboost':
        import xgboost as xgb
        model = xgb.XGBClassifier(**model_params)
    # ...
```

### Novas Métricas
```python
# src/pipeline/evaluator.py
def calculate_metrics(self, y_true, y_pred, y_pred_proba=None):
    # ... métricas existentes ...
    if 'balanced_accuracy' in self.config['evaluation']['metrics']:
        from sklearn.metrics import balanced_accuracy_score
        metrics['balanced_accuracy'] = balanced_accuracy_score(y_true, y_pred)
    # ...
```

### Preprocessadores Customizados
```python
# src/pipeline/data_processor.py
def _apply_custom_preprocessing(self, X_train, X_test):
    # Implementar transformações específicas
    # Normalização específica do domínio
    # Feature engineering automático
    return X_train_processed, X_test_processed
```

## 🚀 Casos de Uso Avançados

### 1. Análise Comparativa de Algoritmos
```python
# Compare múltiplos algoritmos automaticamente
algorithms = ['random_forest', 'logistic_regression', 'svm']
results = compare_algorithms(algorithms, dataset='wine')
best_model = select_best_model(results, metric='f1_score')
```

### 2. Pipeline Automatizado de Produção
```python
# Deploy automatizado
def production_pipeline():
    model = load_best_model()
    new_data = fetch_production_data()
    predictions = model.predict(new_data)
    save_predictions_to_db(predictions)
    monitor_model_drift(model, new_data)
```

### 3. A/B Testing de Modelos  
```python
# Teste diferentes versões
model_a = load_model("model_v1")
model_b = load_model("model_v2")

results_a = evaluate_on_test_set(model_a, test_data)
results_b = evaluate_on_test_set(model_b, test_data)

winner = statistical_test(results_a, results_b)
```

## 🔍 Troubleshooting

### Problemas Comuns

#### "Modelo não encontrado"
```bash
# Verifique se o arquivo existe
ls data/models/
# Execute exemplo de treinamento
python examples/example_usage.py
```

#### "Features faltantes"
```python
# As features do CSV devem coincidir com o modelo
# Verifique nomes das colunas
model_features = manager.get_model_summary("modelo")['features']['names']
print("Features necessárias:", model_features)
```

#### "Erro de memória"
```yaml
# Reduza o dataset ou use algoritmos mais leves
data:
  test_size: 0.1  # Usar menos dados para teste

model:
  algorithm: "logistic_regression"  # Mais leve que RF
```

### Logs Detalhados
```bash
# Verificar logs para debugging
tail -f logs/ml_pipeline_*.log

# Ou via interface web na aba "Logs"
```

## 📈 Roadmap e Melhorias Futuras

### 🔜 Próximas Versões
- [ ] **AutoML**: Seleção automática de algoritmos e hiperparâmetros
- [ ] **Ensemble Methods**: Combinação automática de modelos
- [ ] **Deep Learning**: Integração com redes neurais (TensorFlow/PyTorch)
- [ ] **Time Series**: Suporte a dados temporais
- [ ] **Regressão**: Pipelines para problemas de regressão
- [ ] **Interpretabilidade**: SHAP values e LIME
- [ ] **Deployment**: Export para produção (Docker, FastAPI)
- [ ] **Monitoring**: Drift detection e alertas

### 🌟 Melhorias Planejadas
- [ ] **Interface**: Temas dark/light, mais customização
- [ ] **Performance**: Processamento paralelo, cache de modelos
- [ ] **Dados**: Suporte a mais formatos (Parquet, JSON, SQL)
- [ ] **Colaboração**: Multi-usuário, controle de versão de modelos
- [ ] **Integração**: MLflow, Weights & Biases, cloud providers

## 🤝 Contribuindo

### Como Contribuir
1. 🍴 Fork o projeto
2. 🌿 Crie branch: `git checkout -b feature/nova-funcionalidade`
3. 💻 Implemente suas mudanças
4. ✅ Teste: `python -m pytest tests/`
5. 📝 Commit: `git commit -m 'Adiciona nova funcionalidade'`
6. 🚀 Push: `git push origin feature/nova-funcionalidade`
7. 🔄 Pull Request

### Diretrizes
- **Código**: Siga PEP 8, adicione docstrings
- **Testes**: Cubra novas funcionalidades com testes
- **Documentação**: Atualize README e exemplos
- **Commit**: Mensagens claras e descritivas

## 📝 Licença

Distribuído sob a licença MIT. Veja `LICENSE` para mais informações.

## 📞 Suporte e Contato

- 📧 **Email**: seu-email@exemplo.com
- 🐛 **Issues**: [GitHub Issues](https://github.com/seu-usuario/ml-pipeline-orchestrator/issues)
- 📖 **Documentação**: [Wiki do Projeto](https://github.com/seu-usuario/ml-pipeline-orchestrator/wiki)
- 💬 **Discussões**: [GitHub Discussions](https://github.com/seu-usuario/ml-pipeline-orchestrator/discussions)

## 🙏 Agradecimentos

- **Streamlit**: Interface web fantástica
- **Scikit-learn**: Base sólida para ML
- **Plotly**: Visualizações interativas
- **Comunidade Python**: Ecossistema incrível

---

**Desenvolvido com ❤️ usando Streamlit, Scikit-learn e muito café ☕**

### 🌟 Se este projeto foi útil, considere dar uma estrela no GitHub!

```
⭐ Star this repo | 🍴 Fork | 📢 Share | 🤝 Contribute
```