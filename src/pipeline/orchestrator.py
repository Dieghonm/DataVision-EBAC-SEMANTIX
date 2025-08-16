# src/pipeline/orchestrator.py
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
            data_source = self.config['data']['source']
            if data_source == 'external':
                raise ValueError("Para fonte 'external', é necessário fornecer uploaded_data")
                
            self.data = self.data_processor.load_dataset(data_source)
            self.logger.info(f"📊 Dataset '{data_source}' carregado: {self.data.shape}")
        
        # Validar se os dados foram carregados corretamente
        if self.data is None or self.data.empty:
            raise ValueError("Falha ao carregar dados - dataset vazio ou None")
        
        self.results['data_shape'] = self.data.shape
        self.results['data_info'] = {
            'columns': list(self.data.columns),
            'missing_values': self.data.isnull().sum().to_dict(),
            'dtypes': {col: str(dtype) for col, dtype in self.data.dtypes.to_dict().items()}
        }
    
    def _preprocess_data(self):
        """Preprocessa os dados."""
        self.logger.info("🔧 Preprocessando dados...")
        
        # Separar features e target
        self.X, self.y = self.data_processor.prepare_features_target(self.data)
        
        # Dividir em treino e teste
        self.X_train, self.X_test, self.y_train, self.y_test = \
            self.data_processor.split_data(self.X, self.y)
        
        # Aplicar preprocessamento - CORREÇÃO: passar y_train para seleção de features
        self.X_train_processed, self.X_test_processed = \
            self.data_processor.preprocess_features(
                self.X_train, self.X_test, self.y_train
            )
        
        self.logger.info(f"🎯 Dados divididos - Treino: {self.X_train_processed.shape}, "
                        f"Teste: {self.X_test_processed.shape}")
        
        self.results['preprocessing'] = {
            'train_shape': self.X_train_processed.shape,
            'test_shape': self.X_test_processed.shape,
            'feature_names': self.data_processor.get_feature_names(),
            'original_features': self.X.shape[1],
            'processed_features': self.X_train_processed.shape[1]
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
        
        # Obter informações adicionais do treinamento
        model_info = {
            'algorithm': algorithm,
            'parameters': self.model_trainer.get_model_params(),
            'best_params': self.model_trainer.get_best_params(),
            'cv_scores': self.model_trainer.get_cv_scores()
        }
        
        self.results['model_info'] = model_info
    
    def _evaluate_model(self):
        """Avalia o modelo."""
        self.logger.info("📈 Avaliando modelo...")
        
        # Fazer predições
        self.y_pred = self.model.predict(self.X_test_processed)
        self.y_pred_proba = None
        
        # Tentar obter probabilidades
        if hasattr(self.model, 'predict_proba'):
            try:
                self.y_pred_proba = self.model.predict_proba(self.X_test_processed)
            except Exception as e:
                self.logger.warning(f"Não foi possível obter probabilidades: {str(e)}")
        
        # Calcular métricas
        metrics = self.evaluator.calculate_metrics(
            self.y_test, 
            self.y_pred, 
            self.y_pred_proba
        )
        
        # Gerar gráficos
        try:
            plots = self.evaluator.generate_plots(
                self.y_test, 
                self.y_pred, 
                self.y_pred_proba,
                self.model,
                self.data_processor.get_feature_names()
            )
        except Exception as e:
            self.logger.warning(f"Erro ao gerar gráficos: {str(e)}")
            plots = {}
        
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
            
            # Criar diretório de resultados se não existir
            results_dir = Path(self.config.get('output', {}).get('results_dir', 'data/results'))
            results_dir.mkdir(parents=True, exist_ok=True)
            
            # Salvar modelo (pickle)
            try:
                import joblib
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                model_filename = results_dir / f"model_{self.config['model']['algorithm']}_{timestamp}.pkl"
                joblib.dump(self.model, model_filename)
                self.logger.info(f"Modelo salvo em: {model_filename}")
            except Exception as e:
                self.logger.warning(f"Erro ao salvar modelo: {str(e)}")
            
            # Salvar configuração usada
            try:
                config_filename = results_dir / f"config_{timestamp}.yaml"
                with open(config_filename, 'w', encoding='utf-8') as f:
                    yaml.dump(self.config, f, default_flow_style=False, allow_unicode=True)
                self.logger.info(f"Configuração salva em: {config_filename}")
            except Exception as e:
                self.logger.warning(f"Erro ao salvar configuração: {str(e)}")
    
    def get_results(self):
        """Retorna todos os resultados do pipeline."""
        return self.results
    
    def get_model(self):
        """Retorna o modelo treinado."""
        return self.model
    
    def get_predictions(self):
        """Retorna as predições."""
        return getattr(self, 'y_pred', None), getattr(self, 'y_pred_proba', None)
    
    def get_data_info(self):
        """Retorna informações sobre os dados processados."""
        return {
            'original_data_shape': getattr(self, 'data', pd.DataFrame()).shape,
            'train_shape': getattr(self, 'X_train_processed', pd.DataFrame()).shape,
            'test_shape': getattr(self, 'X_test_processed', pd.DataFrame()).shape,
            'feature_names': self.data_processor.get_feature_names() if self.data_processor else [],
            'target_classes': len(np.unique(getattr(self, 'y', []))) if hasattr(self, 'y') else 0
        }