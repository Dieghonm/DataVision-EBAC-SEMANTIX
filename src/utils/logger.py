# src/utils/logger.py
import logging
import sys
from pathlib import Path
from datetime import datetime

def setup_logger(name, log_file=None, level=logging.INFO):
    """
    Configura sistema de logging.
    
    Args:
        name (str): Nome do logger
        log_file (str): Arquivo de log (opcional)
        level: Nível de logging
        
    Returns:
        Logger configurado
    """
    # Criar diretório de logs se não existir
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    # Nome do arquivo de log baseado na data
    if log_file is None:
        timestamp = datetime.now().strftime("%Y%m%d")
        log_file = log_dir / f"ml_pipeline_{timestamp}.log"
    
    # Configurar formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Configurar logger
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Evitar duplicação de handlers
    if not logger.handlers:
        # Handler para arquivo
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setFormatter(formatter)
        file_handler.setLevel(level)
        
        # Handler para console
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        console_handler.setLevel(level)
        
        # Adicionar handlers
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
    
    return logger

# src/utils/config_loader.py
import yaml
import json
from pathlib import Path
from typing import Dict, Any

class ConfigLoader:
    """
    Classe para carregar e validar configurações.
    """
    
    def __init__(self, config_dir="configs"):
        self.config_dir = Path(config_dir)
        
    def load_config(self, config_name):
        """
        Carrega configuração de arquivo YAML.
        
        Args:
            config_name (str): Nome do arquivo de configuração
            
        Returns:
            dict: Configuração carregada
        """
        if not config_name.endswith('.yaml'):
            config_name += '.yaml'
            
        config_path = self.config_dir / config_name
        
        if not config_path.exists():
            raise FileNotFoundError(f"Configuração não encontrada: {config_path}")
        
        with open(config_path, 'r', encoding='utf-8') as file:
            config = yaml.safe_load(file)
        
        # Validar configuração
        self._validate_config(config)
        
        return config
    
    def save_config(self, config, config_name):
        """
        Salva configuração em arquivo YAML.
        
        Args:
            config (dict): Configuração para salvar
            config_name (str): Nome do arquivo
        """
        if not config_name.endswith('.yaml'):
            config_name += '.yaml'
            
        config_path = self.config_dir / config_name
        
        # Criar diretório se não existir
        self.config_dir.mkdir(exist_ok=True)
        
        with open(config_path, 'w', encoding='utf-8') as file:
            yaml.dump(config, file, default_flow_style=False, allow_unicode=True)
    
    def list_configs(self):
        """
        Lista todas as configurações disponíveis.
        
        Returns:
            list: Lista de nomes de configuração
        """
        if not self.config_dir.exists():
            return []
        
        configs = []
        for file_path in self.config_dir.glob('*.yaml'):
            configs.append(file_path.stem)
        
        return sorted(configs)
    
    def _validate_config(self, config):
        """
        Valida estrutura básica da configuração.
        
        Args:
            config (dict): Configuração para validar
        """
        required_sections = ['project', 'data', 'model', 'pipeline_steps']
        
        for section in required_sections:
            if section not in config:
                raise ValueError(f"Seção obrigatória '{section}' não encontrada na configuração")
        
        # Validar pipeline_steps
        valid_steps = [
            'load_data', 'preprocess_data', 'train_model', 
            'evaluate_model', 'save_results'
        ]
        
        for step in config['pipeline_steps']:
            if step not in valid_steps:
                raise ValueError(f"Etapa inválida: '{step}'. Etapas válidas: {valid_steps}")
        
        # Validar algoritmo
        valid_algorithms = ['random_forest', 'logistic_regression', 'svm']
        algorithm = config['model']['algorithm']
        
        if algorithm not in valid_algorithms:
            raise ValueError(f"Algoritmo inválido: '{algorithm}'. Algoritmos válidos: {valid_algorithms}")

# src/utils/__init__.py
from .logger import setup_logger
from .config_loader import ConfigLoader

__all__ = ['setup_logger', 'ConfigLoader']