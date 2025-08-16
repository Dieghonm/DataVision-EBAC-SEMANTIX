# src/pipeline/data_processor.py
import pandas as pd
import numpy as np
import logging
from sklearn.datasets import load_iris, load_wine, load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.feature_selection import SelectKBest, RFE
from sklearn.ensemble import RandomForestClassifier

class DataProcessor:
    """
    Classe responsável por todo o processamento de dados.
    """
    
    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger('data_processor')
        self.scaler = None
        self.feature_selector = None
        self.feature_names = None
        
    def load_dataset(self, source):
        """
        Carrega dataset baseado na fonte especificada.
        
        Args:
            source (str): Nome do dataset ('iris', 'wine', 'breast_cancer', 'external')
            
        Returns:
            pd.DataFrame: Dataset carregado
        """
        self.logger.info(f"Carregando dataset: {source}")
        
        if source == "iris":
            data = load_iris()
            df = pd.DataFrame(data.data, columns=data.feature_names)
            df['target'] = data.target
            
        elif source == "wine":
            data = load_wine()
            df = pd.DataFrame(data.data, columns=data.feature_names)
            df['target'] = data.target
            
        elif source == "breast_cancer":
            data = load_breast_cancer()
            df = pd.DataFrame(data.data, columns=data.feature_names)
            df['target'] = data.target
            
        elif source == "external":
            # Para datasets externos, retornar None pois serão fornecidos pelo orchestrador
            return None
            
        else:
            raise ValueError(f"Dataset '{source}' não suportado")
        
        self.logger.info(f"Dataset carregado: {df.shape}")
        return df
    
    def prepare_features_target(self, data):
        """
        Separa features e target do dataset.
        
        Args:
            data (pd.DataFrame): Dataset completo
            
        Returns:
            tuple: (X, y) features e target
        """
        if 'target' in data.columns:
            X = data.drop('target', axis=1)
            y = data['target']
        else:
            # Se não há coluna 'target', assume que a última coluna é o target
            X = data.iloc[:, :-1]
            y = data.iloc[:, -1]
        
        self.feature_names = list(X.columns)
        self.logger.info(f"Features: {X.shape[1]}, Target classes: {len(np.unique(y))}")
        
        return X, y
    
    def split_data(self, X, y):
        """
        Divide dados em treino e teste.
        
        Args:
            X: Features
            y: Target
            
        Returns:
            tuple: X_train, X_test, y_train, y_test
        """
        test_size = self.config['data']['test_size']
        random_state = self.config['data']['random_state']
        shuffle = self.config['data'].get('shuffle', True)
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=test_size,
            random_state=random_state,
            shuffle=shuffle,
            stratify=y
        )
        
        self.logger.info(f"Dados divididos - Treino: {X_train.shape}, Teste: {X_test.shape}")
        return X_train, X_test, y_train, y_test
    
    def preprocess_features(self, X_train, X_test, y_train=None):
        """
        Aplica preprocessamento nas features.
        
        Args:
            X_train: Features de treino
            X_test: Features de teste
            y_train: Target de treino (necessário para seleção de features)
            
        Returns:
            tuple: X_train_processed, X_test_processed
        """
        X_train_processed = X_train.copy()
        X_test_processed = X_test.copy()
        
        # 1. Tratar valores faltantes
        X_train_processed, X_test_processed = self._handle_missing_values(
            X_train_processed, X_test_processed
        )
        
        # 2. Aplicar escalonamento
        X_train_processed, X_test_processed = self._apply_scaling(
            X_train_processed, X_test_processed
        )
        
        # 3. Seleção de features (se habilitado) - CORREÇÃO DO BUG
        feature_selection_config = self.config['preprocessing'].get('feature_selection', {})
        if feature_selection_config.get('enabled', False) and y_train is not None:
            X_train_processed, X_test_processed = self._select_features(
                X_train_processed, X_test_processed, y_train
            )
        
        return X_train_processed, X_test_processed
    
    def _handle_missing_values(self, X_train, X_test):
        """Trata valores faltantes."""
        strategy = self.config['preprocessing']['handle_missing']['strategy']
        
        if X_train.isnull().sum().sum() > 0:
            self.logger.info(f"Tratando valores faltantes com estratégia: {strategy}")
            
            if strategy == 'drop':
                X_train = X_train.dropna()
                X_test = X_test.dropna()
            
            elif strategy in ['mean', 'median']:
                from sklearn.impute import SimpleImputer
                imputer = SimpleImputer(strategy=strategy)
                
                X_train = pd.DataFrame(
                    imputer.fit_transform(X_train),
                    columns=X_train.columns,
                    index=X_train.index
                )
                X_test = pd.DataFrame(
                    imputer.transform(X_test),
                    columns=X_test.columns,
                    index=X_test.index
                )
        
        return X_train, X_test
    
    def _apply_scaling(self, X_train, X_test):
        """Aplica escalonamento das features."""
        scaling_method = self.config['preprocessing']['scaling']['method']
        
        if scaling_method == 'none':
            return X_train, X_test
        
        self.logger.info(f"Aplicando escalonamento: {scaling_method}")
        
        if scaling_method == 'standard':
            self.scaler = StandardScaler()
        elif scaling_method == 'minmax':
            feature_range = self.config['preprocessing']['scaling'].get('feature_range', [0, 1])
            self.scaler = MinMaxScaler(feature_range=feature_range)
        elif scaling_method == 'robust':
            self.scaler = RobustScaler()
        else:
            raise ValueError(f"Método de escalonamento '{scaling_method}' não suportado")
        
        X_train_scaled = pd.DataFrame(
            self.scaler.fit_transform(X_train),
            columns=X_train.columns,
            index=X_train.index
        )
        
        X_test_scaled = pd.DataFrame(
            self.scaler.transform(X_test),
            columns=X_test.columns,
            index=X_test.index
        )
        
        return X_train_scaled, X_test_scaled
    
    def _select_features(self, X_train, X_test, y_train):
        """Aplica seleção de features."""
        feature_selection_config = self.config['preprocessing']['feature_selection']
        method = feature_selection_config.get('method', 'selectkbest')
        k = feature_selection_config.get('k', 10)
        
        # Ajustar k se for maior que o número de features
        k = min(k, X_train.shape[1])
        
        self.logger.info(f"Selecionando {k} features com método: {method}")
        
        if method == 'selectkbest':
            from sklearn.feature_selection import f_classif
            self.feature_selector = SelectKBest(score_func=f_classif, k=k)
        elif method == 'rfe':
            estimator = RandomForestClassifier(n_estimators=10, random_state=42)
            self.feature_selector = RFE(estimator, n_features_to_select=k)
        else:
            raise ValueError(f"Método de seleção '{method}' não suportado")
        
        X_train_selected = self.feature_selector.fit_transform(X_train, y_train)
        X_test_selected = self.feature_selector.transform(X_test)
        
        # Atualizar nomes das features selecionadas
        selected_features = np.array(self.feature_names)[self.feature_selector.get_support()]
        self.feature_names = list(selected_features)
        
        # Converter de volta para DataFrame
        X_train_selected = pd.DataFrame(
            X_train_selected,
            columns=self.feature_names,
            index=X_train.index
        )
        
        X_test_selected = pd.DataFrame(
            X_test_selected,
            columns=self.feature_names,
            index=X_test.index
        )
        
        self.logger.info(f"Features selecionadas: {len(self.feature_names)}")
        
        return X_train_selected, X_test_selected
    
    def get_feature_names(self):
        """Retorna os nomes das features após processamento."""
        return self.feature_names
    
    def get_scaler(self):
        """Retorna o scaler treinado."""
        return self.scaler
    
    def get_feature_selector(self):
        """Retorna o seletor de features treinado."""
        return self.feature_selector