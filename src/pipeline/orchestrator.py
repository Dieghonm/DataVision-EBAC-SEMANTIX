# src/pipeline/orchestrator.py
import logging
import yaml
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path

from .data_processor import DataProcessor
from .model_trainer import ModelTrainer
from .evaluator import ModelEvaluator
from ..utils.logger import setup_logger

class MLOrchestrator:
    """
    Orquestrador principal do pipeline de ML.
    Coordena todas as etapas baseado na configuração YAML.
    """
    
    def __init__(self, config_path: str = None, config_dict: dict = None):
        self.config = self._load_config(config_path, config_dict)
        self.logger = setup_logger('orchestrator')
        self.results = {}
        self.data = None
        self.model = None
        
        # Inicializar componentes
        self.data_processor = DataProcessor(self.config)
        self.model_trainer = ModelTrainer(self.config)
        self.evaluator = ModelEvaluator(self.config)
        
    def _load_config(self, config_path: str, config_dict: dict):
        """Carrega configuração do arquivo YAML ou dicionário."""
        if config_dict:
            return config_dict
        elif config_path:
            with open(config_path, 'r', encoding='utf-8') as file:
                return yaml.safe_load(file)
        else:
            raise ValueError("É necessário fornecer config_path ou config_dict")
    
    def run_pipeline(self, uploaded_data=None):
        """
        Executa o pipeline completo baseado na configuração.
        
        Args:
            uploaded_data: DataFrame opcional com dados enviados pelo usuário
        
        Returns:
            dict: Resultados do pipeline
        """
        self.logger.info(f"🚀 Iniciando pipeline: {self.config['project']['name']}")
        start_time = datetime.now()
        
        try:
            # Executar cada etapa configurada
            for step in self.config['pipeline_steps']:
                self.logger.info(f"📋 Executando etapa: {step}")
                
                if step == "load_data":
                    self._load_data(uploaded_data)
                
                elif step == "preprocess_data":
                    self._preprocess_data()
                
                elif step == "train_model":
                    self._train_model()
                
                elif step == "evaluate_model":
                    self._evaluate_model()
                
                elif step == "save_results":
                    self._save_results()
                
                else:
                    self.logger.warning(f"⚠️ Etapa desconhecida: {step}")
            
            # Calcular tempo total
            total_time = datetime.now() - start_time
            self.logger.info(f"✅ Pipeline concluído em {total_time.total_seconds():.2f}s")
            
            self.results['execution_time'] = total_time.total_seconds()
            self.results['status'] = 'success'
            
        except Exception as e:
            self.logger.error(f"❌ Erro no pipeline: {str(e)}")
            self.results['status'] = 'error'
            self.results['error'] = str(e)
            raise
        
        return self.results
    
    def _load_data(self, uploaded_data=None):
        """Carrega dados baseado na configuração."""
        self.logger.info("📂 Carregando dados...")
        
        if uploaded_data is not None:
            self.data = uploaded_data
            self.logger.info(f"📊 Dados carregados do upload: {self.data.shape}")
        else:
            # Carregar datasets padrão
            self.data = self.data_processor.load_dataset(
                self.config['data']['source']
            )
            self.logger.info(f"📊 Dataset '{self.config['data']['source']}' carregado: {self.data.shape}")
        
        self.results['data_shape'] = self.data.shape
        self.results['data_info'] = {
            'columns': list(self.data.columns),
            'missing_values': self.data.isnull().sum().to_dict(),
            'dtypes': self.data.dtypes.to_dict()
        }
    
    def _preprocess_data(self):
        """Preprocessa os dados."""
        self.logger.info("🔧 Preprocessando dados...")
        
        # Separar features e target
        self.X, self.y = self.data_processor.prepare_features_target(self.data)
        
        # Dividir em treino e teste
        self.X_train, self.X_test, self.y_train, self.y_test = \
            self.data_processor.split_data(self.X, self.y)
        
        # Aplicar preprocessamento
        self.X_train_processed, self.X_test_processed = \
            self.data_processor.preprocess_features(self.X_train, self.X_test)
        
        self.logger.info(f"🎯 Dados divididos - Treino: {self.X_train_processed.shape}, "
                        f"Teste: {self.X_test_processed.shape}")
        
        self.results['preprocessing'] = {
            'train_shape': self.X_train_processed.shape,
            'test_shape': self.X_test_processed.shape,
            'feature_names': self.data_processor.get_feature_names()
        }
    
    def _train_model(self):
        """Treina o modelo."""
        self.logger.info("🤖 Treinando modelo...")
        
        self.model = self.model_trainer.train(
            self.X_train_processed, 
            self.y_train
        )
        
        algorithm = self.config['model']['algorithm']
        self.logger.info(f"✨ Modelo {algorithm} treinado com sucesso")
        
        self.results['model_info'] = {
            'algorithm': algorithm,
            'parameters': self.model_trainer.get_model_params()
        }
    
    def _evaluate_model(self):
        """Avalia o modelo."""
        self.logger.info("📈 Avaliando modelo...")
        
        # Fazer predições
        self.y_pred = self.model.predict(self.X_test_processed)
        self.y_pred_proba = None
        
        if hasattr(self.model, 'predict_proba'):
            self.y_pred_proba = self.model.predict_proba(self.X_test_processed)
        
        # Calcular métricas
        metrics = self.evaluator.calculate_metrics(
            self.y_test, 
            self.y_pred, 
            self.y_pred_proba
        )
        
        # Gerar gráficos
        plots = self.evaluator.generate_plots(
            self.y_test, 
            self.y_pred, 
            self.y_pred_proba,
            self.model,
            self.data_processor.get_feature_names()
        )
        
        self.results['evaluation'] = {
            'metrics': metrics,
            'plots': plots
        }
        
        # Log das métricas principais
        for metric, value in metrics.items():
            if isinstance(value, (int, float)):
                self.logger.info(f"📊 {metric}: {value:.4f}")
    
    def _save_results(self):
        """Salva resultados se configurado."""
        if self.config.get('output', {}).get('save_model', False):
            self.logger.info("💾 Salvando resultados...")
            # Implementar salvamento conforme necessário
            pass
    
    def get_results(self):
        """Retorna todos os resultados do pipeline."""
        return self.results
    
    def get_model(self):
        """Retorna o modelo treinado."""
        return self.model
    
    def get_predictions(self):
        """Retorna as predições."""
        return getattr(self, 'y_pred', None), getattr(self, 'y_pred_proba', None)