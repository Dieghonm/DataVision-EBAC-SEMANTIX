#!/usr/bin/env python3

import os
import sys
from pathlib import Path

def create_project_structure():
    """Cria a estrutura de diretórios do projeto."""
    
    directories = [
        "src/pipeline",
        "src/utils", 
        "configs",
        "data/raw",
        "data/processed", 
        "data/results",
        "logs",
        "tests"
    ]
    
    print("📁 Criando estrutura do projeto...")
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"✅ {directory}")
    
    # Criar arquivos __init__.py
    init_files = [
        "src/__init__.py",
        "src/pipeline/__init__.py",
        "src/utils/__init__.py"
    ]
    
    for init_file in init_files:
        Path(init_file).touch()
        print(f"✅ {init_file}")

def create_default_config():
    """Cria configuração padrão."""
    
    config_content = '''# configs/default_config.yaml
project:
  name: "ML Pipeline Demo"
  version: "1.0.0"
  description: "Pipeline configurável para classificação"

data:
  source: "iris"
  test_size: 0.2
  random_state: 42
  shuffle: true

preprocessing:
  scaling:
    method: "standard"
  feature_selection:
    enabled: false
    method: "selectkbest"
    k: 10
  handle_missing:
    strategy: "mean"

model:
  algorithm: "random_forest"
  random_forest:
    n_estimators: 100
    max_depth: 10
    min_samples_split: 2
    random_state: 42

training:
  cross_validation:
    enabled: true
    cv_folds: 5
  hyperparameter_tuning:
    enabled: false
    method: "grid_search"

evaluation:
  metrics:
    - "accuracy"
    - "precision" 
    - "recall"
    - "f1"
  plots:
    - "confusion_matrix"
    - "roc_curve"
    - "feature_importance"

pipeline_steps:
  - "load_data"
  - "preprocess_data"
  - "train_model"
  - "evaluate_model"

output:
  save_model: true
  results_dir: "data/results"
'''
    
    with open("configs/default_config.yaml", "w", encoding='utf-8') as f:
        f.write(config_content)
    
    print("✅ Configuração padrão criada")

def main():
    """Função principal de setup."""
    
    print("🚀 Configurando ML Pipeline Orchestrator...")
    
    # Verificar Python
    if sys.version_info < (3, 8):
        print("❌ Python 3.8+ é necessário")
        sys.exit(1)
    
    # Criar estrutura
    create_project_structure()
    
    # Criar configuração padrão
    create_default_config()
    
    print("\n✅ Setup concluído!")
    print("\n📋 Próximos passos:")
    print("1. pip install -r requirements.txt")
    print("2. cd src && streamlit run app.py")
    print("\n🎯 Acesse: http://localhost:8501")

if __name__ == "__main__":
    main()