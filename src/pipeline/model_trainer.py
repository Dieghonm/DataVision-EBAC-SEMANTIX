# src/pipeline/model_trainer.py
import logging
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score, GridSearchCV, RandomizedSearchCV

class ModelTrainer:
    """
    Classe responsável pelo treinamento de modelos de ML.
    """
    
    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger('model_trainer')
        self.model = None
        self.best_params = None
        
    def train(self, X_train, y_train):
        """
        Treina o modelo baseado na configuração.
        
        Args:
            X_train: Features de treino
            y_train: Target de treino
            
        Returns:
            Modelo treinado
        """
        algorithm = self.config['model']['algorithm']
        self.logger.info(f"Treinando modelo: {algorithm}")
        
        # Inicializar modelo
        self.model = self._initialize_model(algorithm)
        
        # Aplicar hyperparameter tuning se habilitado
        if self.config.get('training', {}).get('hyperparameter_tuning', {}).get('enabled', False):
            self.model = self._tune_hyperparameters(self.model, X_train, y_train)
        
        # Treinar modelo final
        self.model.fit(X_train, y_train)
        
        # Cross-validation se habilitado
        if self.config.get('training', {}).get('cross_validation', {}).get('enabled', False):
            self._perform_cross_validation(X_train, y_train)
        
        self.logger.info("Modelo treinado com sucesso")
        return self.model
    
    def _initialize_model(self, algorithm):
        """
        Inicializa o modelo baseado no algoritmo escolhido.
        
        Args:
            algorithm (str): Nome do algoritmo
            
        Returns:
            Modelo inicializado
        """
        model_params = self.config['model'].get(algorithm, {})
        
        if algorithm == 'random_forest':
            model = RandomForestClassifier(**model_params)
            
        elif algorithm == 'logistic_regression':
            model = LogisticRegression(**model_params)
            
        elif algorithm == 'svm':
            model_params['probability'] = True
            model = SVC(**model_params)
            
        else:
            raise ValueError(f"Algoritmo '{algorithm}' não suportado")
        
        self.logger.info(f"Modelo {algorithm} inicializado com parâmetros: {model_params}")
        return model
    
    def _tune_hyperparameters(self, model, X_train, y_train):
        """
        Realiza tuning de hiperparâmetros.
        
        Args:
            model: Modelo base
            X_train: Features de treino
            y_train: Target de treino
            
        Returns:
            Melhor modelo encontrado
        """
        tuning_config = self.config['training']['hyperparameter_tuning']
        method = tuning_config.get('method', 'grid_search')
        
        self.logger.info(f"Iniciando tuning de hiperparâmetros: {method}")
        
        # Definir espaço de busca baseado no algoritmo
        param_grid = self._get_param_grid()
        
        if method == 'grid_search':
            search = GridSearchCV(
                model,
                param_grid,
                cv=self.config['training']['cross_validation']['cv_folds'],
                scoring='accuracy',
                n_jobs=-1,
                verbose=1
            )
            
        elif method == 'random_search':
            n_iter = tuning_config.get('n_iter', 10)
            search = RandomizedSearchCV(
                model,
                param_grid,
                n_iter=n_iter,
                cv=self.config['training']['cross_validation']['cv_folds'],
                scoring='accuracy',
                n_jobs=-1,
                verbose=1,
                random_state=self.config['model'].get('random_state', 42)
            )
        
        # Executar busca
        search.fit(X_train, y_train)
        
        self.best_params = search.best_params_
        self.logger.info(f"Melhores parâmetros encontrados: {self.best_params}")
        self.logger.info(f"Melhor score: {search.best_score_:.4f}")
        
        return search.best_estimator_
    
    def _get_param_grid(self):
        """
        Define o espaço de busca para cada algoritmo.
        
        Returns:
            dict: Espaço de busca de parâmetros
        """
        algorithm = self.config['model']['algorithm']
        
        if algorithm == 'random_forest':
            param_grid = {
                'n_estimators': [50, 100, 200],
                'max_depth': [5, 10, 15, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            }
            
        elif algorithm == 'logistic_regression':
            param_grid = {
                'C': [0.01, 0.1, 1, 10, 100],
                'solver': ['liblinear', 'lbfgs'],
                'max_iter': [1000, 2000]
            }
            
        elif algorithm == 'svm':
            param_grid = {
                'C': [0.1, 1, 10],
                'kernel': ['rbf', 'linear'],
                'gamma': ['scale', 'auto', 0.001, 0.01]
            }
            
        else:
            param_grid = {}
        
        return param_grid
    
    def _perform_cross_validation(self, X_train, y_train):
        """
        Realiza validação cruzada.
        
        Args:
            X_train: Features de treino
            y_train: Target de treino
        """
        cv_folds = self.config['training']['cross_validation']['cv_folds']
        
        self.logger.info(f"Realizando validação cruzada com {cv_folds} folds")
        
        cv_scores = cross_val_score(
            self.model, 
            X_train, 
            y_train, 
            cv=cv_folds,
            scoring='accuracy',
            n_jobs=-1
        )
        
        mean_score = cv_scores.mean()
        std_score = cv_scores.std()
        
        self.logger.info(f"CV Score: {mean_score:.4f} (+/- {std_score * 2:.4f})")
        
        # Salvar scores para análise posterior
        self.cv_scores = {
            'scores': cv_scores,
            'mean': mean_score,
            'std': std_score
        }
    
    def get_model(self):
        """Retorna o modelo treinado."""
        return self.model
    
    def get_model_params(self):
        """Retorna os parâmetros do modelo."""
        if self.model:
            return self.model.get_params()
        return {}
    
    def get_best_params(self):
        """Retorna os melhores parâmetros encontrados no tuning."""
        return self.best_params
    
    def get_cv_scores(self):
        """Retorna os scores de validação cruzada."""
        return getattr(self, 'cv_scores', None)
    
    def predict(self, X):
        """
        Faz predições com o modelo treinado.
        
        Args:
            X: Features para predição
            
        Returns:
            Predições
        """
        if self.model is None:
            raise ValueError("Modelo não foi treinado ainda")
        
        return self.model.predict(X)
    
    def predict_proba(self, X):
        """
        Faz predições de probabilidade (se disponível).
        
        Args:
            X: Features para predição
            
        Returns:
            Probabilidades das classes
        """
        if self.model is None:
            raise ValueError("Modelo não foi treinado ainda")
        
        if hasattr(self.model, 'predict_proba'):
            return self.model.predict_proba(X)
        else:
            self.logger.warning("Modelo não suporta predict_proba")
            return None