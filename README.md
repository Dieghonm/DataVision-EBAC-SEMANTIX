# 🚀 ML Pipeline Orchestrator

Uma aplicação web interativa para configurar, executar e avaliar pipelines de Machine Learning com interface Streamlit.

## ✨ Características

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
│       └── config_loader.py     # Carregamento de configurações
├── configs/
│   └── default_config.yaml      # Configuração padrão
├── data/
│   ├── raw/                     # Dados brutos
│   ├── processed/               # Dados processados
│   └── results/                 # Resultados e modelos
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

### 1. **Configuração Básica**
- Escolha a fonte de dados (Iris, Wine, Breast Cancer ou upload)
- Selecione o algoritmo de ML
- Configure parâmetros específicos

### 2. **Configuração Avançada**
- Defina estratégia de preprocessamento
- Configure validação cruzada
- Escolha métricas de avaliação

### 3. **Execução**
- Clique em "🚀 Executar Pipeline"
- Acompanhe logs em tempo real
- Visualize resultados nas abas

### 4. **Análise de Resultados**
- **📈 Métricas**: Accuracy, F1-Score, Precision, Recall
- **📊 Visualizações**: Matriz de confusão, curva ROC, importância das features
- **🔧 Configuração**: YAML da configuração utilizada
- **📝 Logs**: Histórico detalhado da execução

## ⚙️ Configuração via YAML

Exemplo de configuração personalizada:

```yaml
# configs/custom_config.yaml
project:
  name: "Meu Pipeline Personalizado"
  version: "1.0.0"

data:
  source: "wine"
  test_size: 0.3
  random_state: 42

preprocessing:
  scaling:
    method: "minmax"
  feature_selection:
    enabled: true
    method: "selectkbest"
    k: 8

model:
  algorithm: "random_forest"
  random_forest:
    n_estimators: 200
    max_depth: 15
    min_samples_split: 5

training:
  cross_validation:
    enabled: true
    cv_folds: 10
  hyperparameter_tuning:
    enabled: true
    method: "grid_search"

evaluation:
  metrics:
    - "accuracy"
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
```

## 🎨 Funcionalidades Principais

### **Orquestração Configurável**
- Pipeline totalmente configurável via YAML
- Etapas modulares e intercambiáveis
- Configuração visual via interface

### **Processamento de Dados**
- Múltiplas estratégias de escalonamento
- Seleção automática de features
- Tratamento de valores faltantes
- Divisão estratificada dos dados

### **Treinamento de Modelos**
- 3 algoritmos principais implementados
- Hyperparameter tuning automático
- Cross-validation integrada
- Logging detalhado do processo

### **Avaliação Abrangente**
- Métricas de classificação completas
- Visualizações interativas
- Recomendações automáticas
- Relatórios estruturados

### **Interface Intuitiva**
- Configuração visual sem código
- Upload de datasets personalizados
- Histórico de execuções
- Visualizações responsivas

## 🔧 Algoritmos Suportados

### **Random Forest**
- N° estimadores configurável
- Controle de profundidade
- Importância de features nativa

### **Logistic Regression** 
- Regularização ajustável
- Múltiplos solvers
- Rápido e interpretável

### **Support Vector Machine (SVM)**
- Múltiplos kernels (RBF, linear, poly)
- Parâmetro C configurável
- Eficaz para dados de alta dimensão

## 📊 Datasets Incluídos

- **Iris**: Classificação de flores (150 amostras, 4 features)
- **Wine**: Classificação de vinhos (178 amostras, 13 features)
- **Breast Cancer**: Diagnóstico médico (569 amostras, 30 features)
- **Upload Personalizado**: Suporte a CSVs customizados

## 🎯 Métricas e Visualizações

### **Métricas**
- Accuracy, Precision, Recall, F1-Score
- ROC-AUC para classificação binária/multiclasse
- Matriz de confusão com percentuais
- Relatório de classificação detalhado

### **Visualizações**
- Matriz de confusão interativa
- Curvas ROC multiclasse
- Gráfico de importância das features
- Curvas Precision-Recall
- Distribuição de probabilidades

## 🚀 Extensões Futuras

- [ ] Suporte a mais algoritmos (XGBoost, Neural Networks)
- [ ] Pipelines de regressão
- [ ] AutoML integrado
- [ ] Deploy de modelos
- [ ] Monitoramento em produção
- [ ] Integração com MLflow
- [ ] Suporte a dados de séries temporais

## 🤝 Contribuindo

1. Fork o projeto
2. Crie uma branch para sua feature (`git checkout -b feature/AmazingFeature`)
3. Commit suas mudanças (`git commit -m 'Add some AmazingFeature'`)
4. Push para a branch (`git push origin feature/AmazingFeature`)
5. Abra um Pull Request

## 📝 Licença

Distribuído sob a licença MIT. Veja `LICENSE` para mais informações.

## 📞 Suporte

- 📧 Email: seu-email@exemplo.com
- 🐛 Issues: [GitHub Issues](https://github.com/seu-usuario/ml-pipeline-orchestrator/issues)
- 📖 Documentação: [Wiki do Projeto](https://github.com/seu-usuario/ml-pipeline-orchestrator/wiki)

---

**Desenvolvido com ❤️ usando Streamlit e Scikit-learn**