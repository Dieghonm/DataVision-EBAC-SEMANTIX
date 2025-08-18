# src/app.py
# src/app.py
import streamlit as st
import yaml
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import logging
from datetime import datetime
from sklearn.datasets import load_iris, load_wine, load_breast_cancer
import joblib
import json

# Imports locais
from src.pipeline.orchestrator import MLOrchestrator
from src.utils.logger import setup_logger

# Configuração da página
st.set_page_config(
    page_title="DataVision EBAC SEMANTIX",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)


class StreamlitMLApp:
    def __init__(self):
        self.logger = setup_logger('streamlit_app')
        self.orchestrator = None
        self.results = {}
        self.current_data = None
        self.models_dir = Path("data/models")
        self.models_dir.mkdir(parents=True, exist_ok=True)
        
    def run(self):
        """Roda a aplicação Streamlit."""
        self._render_header()
        self._render_sidebar()
        self._render_main_content()
    
    def _render_header(self):
        """Renderiza o cabeçalho da aplicação."""
        st.title("📈 DataVision EBAC SEMANTIX")
        st.markdown("---")
        
        # Métricas do header
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Status", "Pronto", delta=None)
        with col2:
            st.metric("Modelos Disponíveis", "5", delta="+1")
        with col3:
            st.metric("Pipelines Executados", len(st.session_state.get('executions', [])))
        with col4:
            saved_models = list(self.models_dir.glob("*.pkl"))
            st.metric("Modelos Salvos", len(saved_models))
    
    def _load_dataset(self, data_source, uploaded_file=None):
        """Carrega o dataset baseado na seleção do usuário."""
        try:
            if data_source == "upload" and uploaded_file is not None:
                df = pd.read_csv(uploaded_file)
                return df, "Dataset Personalizado"
            
            elif data_source == "iris":
                data = load_iris()
                df = pd.DataFrame(data.data, columns=data.feature_names)
                df['target'] = data.target
                return df, "Iris Dataset"
                
            elif data_source == "wine":
                data = load_wine()
                df = pd.DataFrame(data.data, columns=data.feature_names)
                df['target'] = data.target
                return df, "Wine Dataset"
                
            elif data_source == "breast_cancer":
                data = load_breast_cancer()
                df = pd.DataFrame(data.data, columns=data.feature_names)
                df['target'] = data.target
                return df, "Breast Cancer Dataset"
                
            # Para datasets personalizados
            elif data_source == "Credit":
                try:
                    possible_paths = [
                        "data/raw/credit_scoring.ftr",
                        "../data/raw/credit_scoring.ftr", 
                        "credit_scoring.ftr"
                    ]
                    
                    for path in possible_paths:
                        if Path(path).exists():
                            df = pd.read_feather(path)
                            return df, "Credit Scoring Dataset"
                    
                    st.warning("⚠️ Arquivo credit_scoring.ftr não encontrado. Por favor, coloque o arquivo na pasta data/raw/")
                    return None, None
                    
                except Exception as e:
                    st.warning(f"⚠️ Erro ao carregar Credit dataset: {str(e)}")
                    return None, None
                
            elif data_source == "Hipertension":
                try:
                    possible_paths = [
                        "data/raw/hypertension_dataset.csv",
                        "../data/raw/hypertension_dataset.csv",
                        "hypertension_dataset.csv"
                    ]
                    
                    for path in possible_paths:
                        if Path(path).exists():
                            df = pd.read_csv(path)
                            return df, "Hypertension Dataset"
                    
                    st.warning("⚠️ Arquivo hypertension_dataset.csv não encontrado. Por favor, coloque o arquivo na pasta data/raw/")
                    return None, None
                    
                except Exception as e:
                    st.warning(f"⚠️ Erro ao carregar Hypertension dataset: {str(e)}")
                    return None, None
                
            elif data_source == "Phone addiction":
                try:
                    possible_paths = [
                        "data/raw/teen_phone_addiction_dataset.csv",
                        "../data/raw/teen_phone_addiction_dataset.csv",
                        "teen_phone_addiction_dataset.csv"
                    ]
                    
                    for path in possible_paths:
                        if Path(path).exists():
                            df = pd.read_csv(path)
                            return df, "Teen Phone Addiction Dataset"
                    
                    st.warning("⚠️ Arquivo teen_phone_addiction_dataset.csv não encontrado. Por favor, coloque o arquivo na pasta data/raw/")
                    return None, None
                    
                except Exception as e:
                    st.warning(f"⚠️ Erro ao carregar Phone Addiction dataset: {str(e)}")
                    return None, None
                
        except Exception as e:
            st.error(f"Erro ao carregar dataset: {str(e)}")
            return None, None
            
        return None, None
    
    def _render_sidebar(self):
        """Renderiza a barra lateral com configurações."""
        st.sidebar.title("⚙️ Configurações do Pipeline")
        
        # Seletor de modo
        mode = st.sidebar.radio(
            "🎯 Modo de Operação:",
            ["Treinar Novo Modelo", "Usar Modelo Salvo", "Fazer Predições"],
            key="operation_mode"
        )
        
        if mode == "Treinar Novo Modelo":
            self._render_training_sidebar()
        elif mode == "Usar Modelo Salvo":
            self._render_model_loading_sidebar()
        else:  # Fazer Predições
            self._render_prediction_sidebar()
    
    def _render_training_sidebar(self):
        """Renderiza sidebar para treinamento."""
        st.sidebar.subheader("📂 Dados")
        data_source = st.sidebar.selectbox(
            "Fonte de dados:",
            ["upload", "iris", "wine", "breast_cancer", "Credit", "Hipertension", "Phone addiction"],
            help="Escolha um dataset pré-definido ou faça upload",
            key="data_source_select"
        )
        
        uploaded_file = None
        if data_source == "upload":
            uploaded_file = st.sidebar.file_uploader(
                "Upload CSV:", 
                type=['csv'],
                help="Faça upload do seu dataset em CSV",
                key="file_upload"
            )
        
        # Carregar dados quando houver mudança
        if data_source != st.session_state.get('last_data_source') or uploaded_file != st.session_state.get('last_uploaded_file'):
            self.current_data, dataset_name = self._load_dataset(data_source, uploaded_file)
            st.session_state.current_data = self.current_data
            st.session_state.dataset_name = dataset_name
            st.session_state.last_data_source = data_source
            st.session_state.last_uploaded_file = uploaded_file
            if self.current_data is not None:
                st.rerun()
        
        self.current_data = st.session_state.get('current_data')
        
        # Configurações do modelo
        st.sidebar.subheader("🤖 Modelo")
        algorithm = st.sidebar.selectbox(
            "Algoritmo:",
            ["random_forest", "logistic_regression", "svm"],
            help="Escolha o algoritmo de ML"
        )
        
        self._render_algorithm_params(algorithm)
        
        # Configurações de avaliação
        st.sidebar.subheader("📊 Avaliação")
        test_size = st.sidebar.slider("Tamanho do teste:", 0.1, 0.5, 0.2, 0.05)
        cv_folds = st.sidebar.slider("Cross-validation:", 3, 10, 5)
        
        metrics = st.sidebar.multiselect(
            "Métricas:",
            ["accuracy", "precision", "recall", "f1"],
            default=["accuracy", "f1"]
        )
        
        # Opção para salvar modelo
        st.sidebar.subheader("💾 Salvar Modelo")
        save_model = st.sidebar.checkbox("Salvar modelo após treinamento", value=True)
        
        if save_model:
            model_name = st.sidebar.text_input(
                "Nome do modelo:",
                value=f"{algorithm}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                help="Nome para salvar o modelo"
            )
            st.session_state.model_name_to_save = model_name
        
        # Configuração de pipeline
        st.sidebar.subheader("🔧 Pipeline")
        steps = st.sidebar.multiselect(
            "Etapas:",
            ["load_data", "preprocess_data", "train_model", "evaluate_model", "save_results"],
            default=["load_data", "preprocess_data", "train_model", "evaluate_model"]
        )
        
        if save_model and "save_results" not in steps:
            steps.append("save_results")
        
        # Botão para executar
        if self.current_data is not None:
            if st.sidebar.button("🚀 Executar Pipeline", type="primary"):
                self._execute_pipeline(
                    data_source, uploaded_file, algorithm, 
                    test_size, cv_folds, metrics, steps, save_model
                )
        else:
            st.sidebar.info("👆 Selecione um dataset para continuar")
    
    def _render_model_loading_sidebar(self):
        """Renderiza sidebar para carregar modelo salvo."""
        st.sidebar.subheader("📁 Modelos Salvos")
        
        # Listar modelos disponíveis
        saved_models = list(self.models_dir.glob("*.pkl"))
        
        if not saved_models:
            st.sidebar.warning("Nenhum modelo salvo encontrado.")
            return
        
        model_names = [f.stem for f in saved_models]
        selected_model = st.sidebar.selectbox(
            "Selecione o modelo:",
            model_names,
            help="Escolha um modelo salvo para carregar"
        )
        
        if st.sidebar.button("🔄 Carregar Modelo", type="primary"):
            self._load_saved_model(selected_model)
        
        # Mostrar informações do modelo se carregado
        if hasattr(st.session_state, 'loaded_model_info'):
            with st.sidebar.expander("ℹ️ Informações do Modelo"):
                info = st.session_state.loaded_model_info
                st.write(f"**Algoritmo:** {info.get('algorithm', 'N/A')}")
                st.write(f"**Data:** {info.get('timestamp', 'N/A')}")
                st.write(f"**Acurácia:** {info.get('accuracy', 'N/A'):.4f}")
    
    def _render_prediction_sidebar(self):
        """Renderiza sidebar para fazer predições."""
        st.sidebar.subheader("🔮 Predições")
        
        if not hasattr(st.session_state, 'loaded_model'):
            st.sidebar.warning("⚠️ Carregue um modelo primeiro!")
            st.sidebar.info("Use o modo 'Usar Modelo Salvo' para carregar um modelo.")
            return
        
        # Opções de input
        input_method = st.sidebar.radio(
            "Método de entrada:",
            ["Manual", "Upload CSV"],
            help="Como você quer inserir os dados para predição"
        )
        
        if input_method == "Manual":
            self._render_manual_input()
        else:
            self._render_csv_upload_prediction()
    
    def _render_manual_input(self):
        """Renderiza entrada manual de dados."""
        st.sidebar.subheader("📝 Entrada Manual")
        
        if not hasattr(st.session_state, 'model_feature_names'):
            st.sidebar.error("Informações das features não disponíveis.")
            return
        
        feature_names = st.session_state.model_feature_names
        feature_values = {}
        
        for feature in feature_names:
            # Para simplificar, usar number_input para todas as features
            feature_values[feature] = st.sidebar.number_input(
                f"{feature}:",
                value=0.0,
                help=f"Valor para a feature {feature}"
            )
        
        if st.sidebar.button("🎯 Fazer Predição", type="primary"):
            self._make_single_prediction(feature_values)
    
    def _render_csv_upload_prediction(self):
        """Renderiza upload de CSV para predições."""
        st.sidebar.subheader("📄 Upload CSV")
        
        prediction_file = st.sidebar.file_uploader(
            "Upload dados para predição:",
            type=['csv'],
            help="CSV com as mesmas features do modelo treinado"
        )
        
        if prediction_file and st.sidebar.button("🎯 Fazer Predições", type="primary"):
            self._make_batch_predictions(prediction_file)
    
    def _load_saved_model(self, model_name):
        """Carrega um modelo salvo."""
        try:
            model_path = self.models_dir / f"{model_name}.pkl"
            info_path = self.models_dir / f"{model_name}_info.json"
            
            # Carregar modelo
            model = joblib.load(model_path)
            st.session_state.loaded_model = model
            
            # Carregar informações se disponível
            if info_path.exists():
                with open(info_path, 'r') as f:
                    info = json.load(f)
                st.session_state.loaded_model_info = info
                st.session_state.model_feature_names = info.get('feature_names', [])
                st.session_state.model_scaler = info.get('scaler_path')
            
            st.success(f"✅ Modelo '{model_name}' carregado com sucesso!")
            st.rerun()
            
        except Exception as e:
            st.error(f"❌ Erro ao carregar modelo: {str(e)}")
    
    def _make_single_prediction(self, feature_values):
        """Faz predição com entrada manual."""
        try:
            # Preparar dados
            input_df = pd.DataFrame([feature_values])
            
            # Aplicar scaler se disponível
            if hasattr(st.session_state, 'model_scaler') and st.session_state.model_scaler:
                scaler_path = Path(st.session_state.model_scaler)
                if scaler_path.exists():
                    scaler = joblib.load(scaler_path)
                    input_df = pd.DataFrame(
                        scaler.transform(input_df),
                        columns=input_df.columns
                    )
            
            # Fazer predição
            model = st.session_state.loaded_model
            prediction = model.predict(input_df)[0]
            
            # Tentar obter probabilidades
            if hasattr(model, 'predict_proba'):
                probabilities = model.predict_proba(input_df)[0]
                st.session_state.single_prediction = {
                    'prediction': prediction,
                    'probabilities': probabilities,
                    'input_data': feature_values
                }
            else:
                st.session_state.single_prediction = {
                    'prediction': prediction,
                    'input_data': feature_values
                }
            
            st.success("✅ Predição realizada!")
            
        except Exception as e:
            st.error(f"❌ Erro na predição: {str(e)}")
    
    def _make_batch_predictions(self, uploaded_file):
        """Faz predições em lote com CSV."""
        try:
            # Carregar dados
            input_df = pd.read_csv(uploaded_file)
            
            # Aplicar pré-processamento se necessário
            processed_df = self._preprocess_data_for_ml(input_df)
            
            if processed_df is None:
                st.error("❌ Erro no pré-processamento dos dados")
                return
            
            # Remover coluna target se existir
            if 'target' in processed_df.columns:
                processed_df = processed_df.drop('target', axis=1)
            
            # Fazer predições
            model = st.session_state.loaded_model
            predictions = model.predict(processed_df)
            
            # Tentar obter probabilidades
            if hasattr(model, 'predict_proba'):
                probabilities = model.predict_proba(processed_df)
                st.session_state.batch_predictions = {
                    'predictions': predictions,
                    'probabilities': probabilities,
                    'input_data': input_df,
                    'processed_data': processed_df
                }
            else:
                st.session_state.batch_predictions = {
                    'predictions': predictions,
                    'input_data': input_df,
                    'processed_data': processed_df
                }
            
            st.success(f"✅ {len(predictions)} predições realizadas!")
            
        except Exception as e:
            st.error(f"❌ Erro nas predições em lote: {str(e)}")
    
    def _render_algorithm_params(self, algorithm):
        """Renderiza parâmetros específicos do algoritmo."""
        st.sidebar.subheader(f"Parâmetros - {algorithm.replace('_', ' ').title()}")
        
        if algorithm == "random_forest":
            n_estimators = st.sidebar.slider("N° Estimadores:", 10, 500, 100, 10)
            max_depth = st.sidebar.slider("Profundidade Max:", 1, 30, 10)
            st.session_state.model_params = {
                'n_estimators': n_estimators,
                'max_depth': max_depth
            }
            
        elif algorithm == "logistic_regression":
            C = st.sidebar.slider("Regularização C:", 0.01, 10.0, 1.0, 0.01)
            max_iter = st.sidebar.slider("Max Iterações:", 100, 5000, 1000, 100)
            st.session_state.model_params = {
                'C': C,
                'max_iter': max_iter
            }
            
        elif algorithm == "svm":
            C = st.sidebar.slider("C:", 0.01, 10.0, 1.0, 0.01)
            kernel = st.sidebar.selectbox("Kernel:", ["rbf", "linear", "poly"])
            st.session_state.model_params = {
                'C': C,
                'kernel': kernel
            }
    
    def _preprocess_data_for_ml(self, df):
        """Pré-processa os dados para ML, lidando com tipos problemáticos."""
        if df is None:
            return None
            
        df_processed = df.copy()
        
        # Converter colunas de datetime para features numéricas
        datetime_cols = df_processed.select_dtypes(include=['datetime64', 'datetime']).columns
        for col in datetime_cols:
            df_processed[f'{col}_year'] = pd.to_datetime(df_processed[col]).dt.year
            df_processed[f'{col}_month'] = pd.to_datetime(df_processed[col]).dt.month
            df_processed[f'{col}_day'] = pd.to_datetime(df_processed[col]).dt.day
            df_processed[f'{col}_dayofweek'] = pd.to_datetime(df_processed[col]).dt.dayofweek
            df_processed.drop(col, axis=1, inplace=True)
        
        # Detectar e converter colunas que podem ser Timestamp como objeto
        for col in df_processed.columns:
            if df_processed[col].dtype == 'object':
                try:
                    pd.to_datetime(df_processed[col])
                    df_processed[f'{col}_year'] = pd.to_datetime(df_processed[col]).dt.year
                    df_processed[f'{col}_month'] = pd.to_datetime(df_processed[col]).dt.month
                    df_processed[f'{col}_day'] = pd.to_datetime(df_processed[col]).dt.day
                    df_processed[f'{col}_dayofweek'] = pd.to_datetime(df_processed[col]).dt.dayofweek
                    df_processed.drop(col, axis=1, inplace=True)
                except:
                    if df_processed[col].nunique() < 50:
                        df_processed[col] = pd.Categorical(df_processed[col]).codes
        
        # Garantir que todas as colunas sejam numéricas (exceto target se existir)
        non_numeric_cols = []
        for col in df_processed.columns:
            if col != 'target' and not pd.api.types.is_numeric_dtype(df_processed[col]):
                non_numeric_cols.append(col)
        
        if non_numeric_cols:
            df_processed.drop(non_numeric_cols, axis=1, inplace=True)
        
        # Tratar valores infinitos
        df_processed.replace([np.inf, -np.inf], np.nan, inplace=True)
        
        # Verificar se existe coluna target
        if 'target' not in df_processed.columns:
            target_candidates = [
                'label', 'class', 'y', 'outcome', 'result', 'prediction',
                'classification', 'category', 'grupo', 'classe'
            ]
            
            target_col = None
            for candidate in target_candidates:
                matching_cols = [col for col in df_processed.columns if candidate.lower() in col.lower()]
                if matching_cols:
                    target_col = matching_cols[0]
                    break
            
            if target_col:
                df_processed['target'] = df_processed[target_col]
                df_processed.drop(target_col, axis=1, inplace=True)
            else:
                last_col = df_processed.columns[-1]
                df_processed['target'] = df_processed[last_col]
                df_processed.drop(last_col, axis=1, inplace=True)
        
        return df_processed
    
    def _execute_pipeline(self, data_source, uploaded_file, algorithm, test_size, cv_folds, metrics, steps, save_model):
        """Executa o pipeline com as configurações especificadas."""
        try:
            with st.spinner("🔄 Executando pipeline..."):
                # Criar configuração dinâmica
                config = self._create_config(
                    data_source, algorithm, test_size, cv_folds, metrics, steps
                )
                
                # Determinar que dados usar e pré-processar
                if data_source == "upload" and uploaded_file is not None:
                    pipeline_data = self._preprocess_data_for_ml(self.current_data)
                elif data_source in ["Credit", "Hipertension", "Phone addiction"]:
                    pipeline_data = self._preprocess_data_for_ml(self.current_data)
                    config['data']['source'] = 'external'
                else:
                    pipeline_data = None
                
                # Inicializar e executar orquestrador
                self.orchestrator = MLOrchestrator(config_dict=config)
                self.results = self.orchestrator.run_pipeline(pipeline_data)
                
                # Salvar modelo se solicitado
                if save_model and self.results.get('status') == 'success':
                    self._save_trained_model()
                
                # Salvar na sessão
                if 'executions' not in st.session_state:
                    st.session_state.executions = []
                
                execution = {
                    'timestamp': datetime.now(),
                    'config': config,
                    'results': self.results
                }
                st.session_state.executions.append(execution)
                
                st.success("✅ Pipeline executado com sucesso!")
                st.rerun()
                
        except Exception as e:
            st.error(f"❌ Erro ao executar pipeline: {str(e)}")
            self.logger.error(f"Erro no pipeline: {str(e)}")
    
    def _save_trained_model(self):
        """Salva o modelo treinado com informações adicionais."""
        try:
            model_name = st.session_state.get('model_name_to_save', f"model_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
            
            # Salvar modelo
            model_path = self.models_dir / f"{model_name}.pkl"
            joblib.dump(self.orchestrator.get_model(), model_path)
            
            # Salvar scaler se disponível
            scaler = self.orchestrator.data_processor.get_scaler()
            scaler_path = None
            if scaler:
                scaler_path = self.models_dir / f"{model_name}_scaler.pkl"
                joblib.dump(scaler, scaler_path)
            
            # Salvar informações do modelo
            model_info = {
                'algorithm': self.orchestrator.config['model']['algorithm'],
                'timestamp': datetime.now().isoformat(),
                'feature_names': self.orchestrator.data_processor.get_feature_names(),
                'accuracy': self.results['evaluation']['metrics'].get('accuracy', 0),
                'f1_score': self.results['evaluation']['metrics'].get('f1', 0),
                'scaler_path': str(scaler_path) if scaler_path else None,
                'config': self.orchestrator.config
            }
            
            info_path = self.models_dir / f"{model_name}_info.json"
            with open(info_path, 'w') as f:
                json.dump(model_info, f, indent=2, default=str)
            
            st.success(f"💾 Modelo salvo como '{model_name}'")
            
        except Exception as e:
            st.error(f"❌ Erro ao salvar modelo: {str(e)}")
    
    def _create_config(self, data_source, algorithm, test_size, cv_folds, metrics, steps):
        """Cria configuração baseada nos inputs da interface."""
        model_params = st.session_state.get('model_params', {})
        
        data_source_mapping = {
            'Credit': 'external',
            'Hipertension': 'external', 
            'Phone addiction': 'external'
        }
        
        config_data_source = data_source_mapping.get(data_source, data_source)
        
        config = {
            'project': {
                'name': f'Pipeline {algorithm} - {data_source}',
                'version': '1.0.0'
            },
            'data': {
                'source': config_data_source,
                'test_size': test_size,
                'random_state': 42
            },
            'preprocessing': {
                'scaling': {'method': 'standard'},
                'handle_missing': {'strategy': 'mean'}
            },
            'model': {
                'algorithm': algorithm,
                algorithm: model_params
            },
            'training': {
                'cross_validation': {
                    'enabled': True,
                    'cv_folds': cv_folds
                }
            },
            'evaluation': {
                'metrics': metrics,
                'plots': ['confusion_matrix', 'roc_curve']
            },
            'pipeline_steps': steps,
            'output': {
                'save_model': True,
                'results_dir': str(self.models_dir)
            }
        }
        
        return config
    
    def _render_main_content(self):
        """Renderiza o conteúdo principal."""
        mode = st.session_state.get('operation_mode', 'Treinar Novo Modelo')
        
        if mode == "Fazer Predições":
            self._render_predictions_results()
        elif hasattr(self, 'results') and self.results:
            self._render_results()
        elif self.current_data is not None:
            self._render_data_preview()
        else:
            self._render_welcome()
    
    def _render_predictions_results(self):
        """Renderiza resultados de predições."""
        st.header("🔮 Resultados das Predições")
        
        # Predição única
        if hasattr(st.session_state, 'single_prediction'):
            pred = st.session_state.single_prediction
            
            st.subheader("🎯 Predição Individual")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric("Predição", f"Classe {pred['prediction']}")
                
                # Mostrar probabilidades se disponível
                if 'probabilities' in pred:
                    st.subheader("📊 Probabilidades por Classe")
                    prob_df = pd.DataFrame({
                        'Classe': range(len(pred['probabilities'])),
                        'Probabilidade': pred['probabilities']
                    })
                    
                    fig = px.bar(
                        prob_df, 
                        x='Classe', 
                        y='Probabilidade',
                        title="Probabilidades de Cada Classe"
                    )
                    st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                st.subheader("📝 Dados de Entrada")
                input_df = pd.DataFrame([pred['input_data']])
                st.dataframe(input_df.T, use_container_width=True)
        
        # Predições em lote
        if hasattr(st.session_state, 'batch_predictions'):
            pred_batch = st.session_state.batch_predictions
            
            st.subheader("📊 Predições em Lote")
            
            # Métricas gerais
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total de Predições", len(pred_batch['predictions']))
            with col2:
                unique_classes = len(np.unique(pred_batch['predictions']))
                st.metric("Classes Preditas", unique_classes)
            with col3:
                most_common = np.bincount(pred_batch['predictions']).argmax()
                st.metric("Classe Mais Frequente", most_common)
            
            # Tabela com resultados
            st.subheader("📋 Resultados Detalhados")
            results_df = pred_batch['input_data'].copy()
            results_df['Predição'] = pred_batch['predictions']
            
            # Adicionar probabilidades se disponível
            if 'probabilities' in pred_batch:
                probs = pred_batch['probabilities']
                for i in range(probs.shape[1]):
                    results_df[f'Prob_Classe_{i}'] = probs[:, i]
            
            st.dataframe(results_df, use_container_width=True)
            
            # Gráfico de distribuição das predições
            st.subheader("📈 Distribuição das Predições")
            pred_counts = pd.Series(pred_batch['predictions']).value_counts().reset_index()
            pred_counts.columns = ['Classe', 'Quantidade']
            
            fig = px.pie(
                pred_counts, 
                values='Quantidade', 
                names='Classe',
                title="Distribuição das Classes Preditas"
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Opção para download
            csv = results_df.to_csv(index=False)
            st.download_button(
                label="📥 Download Resultados (CSV)",
                data=csv,
                file_name=f"predicoes_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
        
        if not hasattr(st.session_state, 'single_prediction') and not hasattr(st.session_state, 'batch_predictions'):
            st.info("👆 Use a barra lateral para fazer predições com um modelo carregado.")
    
    def _render_results(self):
        """Renderiza os resultados do pipeline."""
        st.header("📊 Resultados do Pipeline")
        
        # Status e métricas gerais no topo
        self._render_pipeline_metrics()
        
        # Abas para diferentes seções
        tab1, tab2, tab3, tab4 = st.tabs(["📈 Métricas", "📊 Visualizações", "🔧 Configuração", "📝 Logs"])
        
        with tab1:
            self._render_metrics_tab()
            
        with tab2:
            self._render_visualizations_tab()
            
        with tab3:
            self._render_config_tab()
            
        with tab4:
            self._render_logs_tab()
    
    def _render_pipeline_metrics(self):
        """Renderiza métricas principais do pipeline de forma destacada."""
        # Métricas de status
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            status = self.results.get('status', 'unknown')
            if status == 'success':
                st.success("✅ Sucesso")
            else:
                st.error("❌ Erro")
            
        with col2:
            exec_time = self.results.get('execution_time', 0)
            st.metric("⏱️ Tempo de Execução", f"{exec_time:.2f}s")
            
        with col3:
            if 'data_shape' in self.results:
                shape = self.results['data_shape']
                st.metric("📊 Amostras", shape[0])
                
        with col4:
            if 'data_shape' in self.results:
                shape = self.results['data_shape']
                st.metric("🔢 Features", shape[1])
        
        # Métricas de Performance - DESTACADAS
        if 'evaluation' in self.results:
            st.markdown("### 🎯 Performance do Modelo")
            
            metrics = self.results['evaluation']['metrics']
            
            # Métricas principais em destaque
            metric_cols = st.columns(len([k for k, v in metrics.items() if isinstance(v, (int, float)) and k != 'roc_auc']))
            
            col_idx = 0
            for metric, value in metrics.items():
                if isinstance(value, (int, float)) and metric != 'roc_auc':  # Mostrar ROC-AUC separadamente se existir
                    with metric_cols[col_idx]:
                        # Definir cor baseada na performance
                        if value >= 0.9:
                            delta_color = "normal"
                            delta = "Excelente"
                        elif value >= 0.8:
                            delta_color = "normal" 
                            delta = "Bom"
                        elif value >= 0.7:
                            delta_color = "off"
                            delta = "Regular"
                        else:
                            delta_color = "inverse"
                            delta = "Precisa Melhorar"
                        
                        st.metric(
                            label=metric.replace('_', ' ').title(), 
                            value=f"{value:.4f}",
                            delta=delta
                        )
                    col_idx += 1
            
            # ROC-AUC em destaque se disponível
            if 'roc_auc' in metrics:
                st.markdown("#### 📈 Área Sob a Curva ROC (AUC)")
                roc_auc = metrics['roc_auc']
                col1, col2, col3 = st.columns([1, 2, 1])
                with col2:
                    # Gauge chart para AUC
                    fig = go.Figure(go.Indicator(
                        mode = "gauge+number+delta",
                        value = roc_auc,
                        domain = {'x': [0, 1], 'y': [0, 1]},
                        title = {'text': "ROC-AUC Score"},
                        delta = {'reference': 0.5},
                        gauge = {
                            'axis': {'range': [None, 1]},
                            'bar': {'color': "darkblue"},
                            'steps': [
                                {'range': [0, 0.5], 'color': "lightgray"},
                                {'range': [0.5, 0.7], 'color': "yellow"},
                                {'range': [0.7, 0.9], 'color': "orange"},
                                {'range': [0.9, 1], 'color': "green"}
                            ],
                            'threshold': {
                                'line': {'color': "red", 'width': 4},
                                'thickness': 0.75,
                                'value': 0.9
                            }
                        }
                    ))
                    fig.update_layout(height=300)
                    st.plotly_chart(fig, use_container_width=True)
            
            # Recomendações baseadas nas métricas
            if hasattr(self, 'orchestrator') and self.orchestrator:
                recommendations = self.orchestrator.evaluator._generate_recommendations(metrics)
                if recommendations:
                    st.markdown("### 💡 Recomendações")
                    for rec in recommendations:
                        if "✅" in rec:
                            st.success(rec)
                        elif "⚠️" in rec:
                            st.warning(rec)
                        else:
                            st.info(rec)
    
    def _render_data_preview(self):
        """Renderiza a prévia dos dados carregados."""
        st.header(f"📊 Dataset: {st.session_state.get('dataset_name', 'Dados Carregados')}")
        
        # Informações básicas do dataset
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("📏 Linhas", self.current_data.shape[0])
        with col2:
            st.metric("📊 Colunas", self.current_data.shape[1])
        with col3:
            st.metric("💾 Memória (KB)", f"{self.current_data.memory_usage(deep=True).sum() / 1024:.1f}")
        with col4:
            missing_values = self.current_data.isnull().sum().sum()
            st.metric("❓ Valores Faltantes", missing_values)
        
        # Abas para diferentes visualizações
        tab1, tab2, tab3, tab4 = st.tabs(["🔍 Dados", "📈 Estatísticas", "📊 Visualizações", "🔍 Análise"])
        
        with tab1:
            st.subheader("Primeiras Linhas")
            st.dataframe(self.current_data.head(5), use_container_width=True)
            
            st.subheader("Informações das Colunas")
            col_info = []
            for col in self.current_data.columns:
                col_info.append({
                    'Coluna': col,
                    'Tipo': str(self.current_data[col].dtype),
                    'Valores Únicos': self.current_data[col].nunique(),
                    'Valores Faltantes': self.current_data[col].isnull().sum(),
                    '% Faltantes': f"{(self.current_data[col].isnull().sum() / len(self.current_data) * 100):.2f}%"
                })
            st.dataframe(pd.DataFrame(col_info), use_container_width=True)
        
        with tab2:
            st.subheader("Estatísticas Descritivas")
            numeric_cols = self.current_data.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                st.dataframe(self.current_data[numeric_cols].describe(), use_container_width=True)
            else:
                st.info("Nenhuma coluna numérica encontrada para estatísticas.")
        
        with tab3:
            self._render_data_visualizations()
        
        with tab4:
            st.subheader("Análise de Qualidade dos Dados")
            
            # Verificar valores faltantes
            missing_data = self.current_data.isnull().sum()
            if missing_data.any():
                st.warning("⚠️ Valores faltantes encontrados:")
                missing_df = pd.DataFrame({
                    'Coluna': missing_data.index,
                    'Valores Faltantes': missing_data.values,
                    '% do Total': (missing_data.values / len(self.current_data) * 100).round(2)
                }).query('`Valores Faltantes` > 0')
                st.dataframe(missing_df, use_container_width=True)
            else:
                st.success("✅ Nenhum valor faltante encontrado!")
            
            # Verificar duplicatas
            duplicates = self.current_data.duplicated().sum()
            if duplicates > 0:
                st.warning(f"⚠️ {duplicates} linhas duplicadas encontradas")
            else:
                st.success("✅ Nenhuma linha duplicada encontrada!")
    
    def _render_data_visualizations(self):
        """Renderiza visualizações dos dados."""
        st.subheader("Visualizações dos Dados")
        
        numeric_cols = self.current_data.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = self.current_data.select_dtypes(include=['object', 'category']).columns.tolist()
        
        if len(numeric_cols) > 0:
            # Histogramas
            st.write("**Distribuições das Variáveis Numéricas**")
            selected_cols = st.multiselect(
                "Selecione colunas para visualizar:",
                numeric_cols,
                default=numeric_cols[:4] if len(numeric_cols) > 4 else numeric_cols
            )
            
            if selected_cols:
                cols_per_row = 2
                rows = len(selected_cols) // cols_per_row + (1 if len(selected_cols) % cols_per_row else 0)
                
                for i in range(0, len(selected_cols), cols_per_row):
                    cols = st.columns(cols_per_row)
                    for j, col_name in enumerate(selected_cols[i:i+cols_per_row]):
                        with cols[j]:
                            fig = px.histogram(
                                self.current_data, 
                                x=col_name, 
                                title=f'Distribuição - {col_name}',
                                height=300
                            )
                            st.plotly_chart(fig, use_container_width=True)
            
            # Matriz de correlação
            if len(numeric_cols) > 1:
                st.write("**Matriz de Correlação**")
                corr_matrix = self.current_data[numeric_cols].corr()
                fig = px.imshow(
                    corr_matrix,
                    text_auto=True,
                    aspect="auto",
                    title="Correlação entre Variáveis Numéricas",
                    color_continuous_scale='RdBu'
                )
                st.plotly_chart(fig, use_container_width=True)
        
        # Gráficos para variáveis categóricas
        if categorical_cols:
            st.write("**Distribuições das Variáveis Categóricas**")
            for col in categorical_cols[:3]:  # Limitar a 3 para não sobrecarregar
                fig = px.bar(
                    x=self.current_data[col].value_counts().values,
                    y=self.current_data[col].value_counts().index,
                    title=f'Distribuição - {col}',
                    orientation='h'
                )
                st.plotly_chart(fig, use_container_width=True)
    
    def _render_welcome(self):
        """Renderiza tela de boas-vindas."""
        st.markdown("""
        ## Bem-vindo ao ML Pipeline Orchestrator!
        
        Esta aplicação permite que você configure e execute pipelines de Machine Learning 
        de forma visual e interativa.
        
        ### 🆕 Novas Funcionalidades:
        - **💾 Salvar Modelos**: Treine e salve seus modelos para reutilização
        - **🔄 Carregar Modelos**: Carregue modelos salvos anteriormente  
        - **🔮 Fazer Predições**: Use modelos treinados para predições em novos dados
        - **📊 Métricas Destacadas**: Visualize a performance dos modelos de forma clara
        
        ### Como usar:
        1. **Escolha o modo** na barra lateral: Treinar, Carregar ou Predizer
        2. **Configure seu pipeline** ou carregue um modelo existente
        3. **Visualize métricas** e resultados de forma interativa
        4. **Faça predições** com novos dados usando modelos treinados
        
        **Comece selecionando um modo de operação na barra lateral!**
        """)
        
        # Seção de Modelos Salvos
        saved_models = list(self.models_dir.glob("*.pkl"))
        if saved_models:
            st.markdown("---")
            st.subheader("💾 Modelos Salvos Disponíveis")
            
            models_data = []
            for model_file in saved_models:
                info_file = model_file.parent / f"{model_file.stem}_info.json"
                if info_file.exists():
                    try:
                        with open(info_file, 'r') as f:
                            info = json.load(f)
                        models_data.append({
                            'Nome': model_file.stem,
                            'Algoritmo': info.get('algorithm', 'N/A'),
                            'Acurácia': f"{info.get('accuracy', 0):.4f}",
                            'F1-Score': f"{info.get('f1_score', 0):.4f}",
                            'Data': info.get('timestamp', 'N/A')[:10] if info.get('timestamp') else 'N/A'
                        })
                    except:
                        models_data.append({
                            'Nome': model_file.stem,
                            'Algoritmo': 'N/A',
                            'Acurácia': 'N/A',
                            'F1-Score': 'N/A',
                            'Data': 'N/A'
                        })
            
            if models_data:
                st.dataframe(pd.DataFrame(models_data), use_container_width=True)
        
        # Histórico de execuções
        if 'executions' in st.session_state and st.session_state.executions:
            st.markdown("---")
            st.subheader("📜 Histórico Recente")
            
            for i, execution in enumerate(reversed(st.session_state.executions[-3:])):
                with st.expander(f"Execução {len(st.session_state.executions) - i} - {execution['timestamp'].strftime('%H:%M:%S')}"):
                    col1, col2 = st.columns(2)
                    with col1:
                        st.write("**Algoritmo:**", execution['config']['model']['algorithm'])
                        st.write("**Status:**", execution['results'].get('status', 'unknown'))
                    with col2:
                        if 'evaluation' in execution['results']:
                            metrics = execution['results']['evaluation']['metrics']
                            for metric, value in metrics.items():
                                if isinstance(value, (int, float)):
                                    st.metric(metric.title(), f"{value:.4f}")
    
    def _render_metrics_tab(self):
        """Renderiza a aba de métricas."""
        if 'evaluation' in self.results:
            metrics = self.results['evaluation']['metrics']
            
            st.subheader("🎯 Métricas de Performance Detalhadas")
            
            # Matriz de confusão se disponível
            if 'confusion_matrix' in metrics:
                st.subheader("🎭 Matriz de Confusão")
                cm = metrics['confusion_matrix']
                
                fig = px.imshow(
                    cm,
                    text_auto=True,
                    aspect="auto",
                    title="Matriz de Confusão"
                )
                st.plotly_chart(fig, use_container_width=True)
            
            # Relatório de classificação
            if 'classification_report' in metrics:
                st.subheader("📋 Relatório de Classificação")
                report = metrics['classification_report']
                
                # Converter para DataFrame para melhor visualização
                report_df = pd.DataFrame(report).transpose()
                
                # Remover linhas que não são classes
                classes_df = report_df[~report_df.index.isin(['accuracy', 'macro avg', 'weighted avg'])]
                if not classes_df.empty:
                    st.dataframe(classes_df, use_container_width=True)
                
                # Mostrar médias
                if 'weighted avg' in report:
                    st.subheader("📊 Médias Ponderadas")
                    weighted_avg = report['weighted avg']
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Precision", f"{weighted_avg['precision']:.4f}")
                    with col2:
                        st.metric("Recall", f"{weighted_avg['recall']:.4f}")
                    with col3:
                        st.metric("F1-Score", f"{weighted_avg['f1-score']:.4f}")
    
    def _render_visualizations_tab(self):
        """Renderiza a aba de visualizações."""
        if 'evaluation' in self.results and 'plots' in self.results['evaluation']:
            plots = self.results['evaluation']['plots']
            
            for plot_name, plot_data in plots.items():
                st.subheader(f"📊 {plot_name.replace('_', ' ').title()}")
                if plot_data:
                    st.plotly_chart(plot_data, use_container_width=True)
    
    def _render_config_tab(self):
        """Renderiza a aba de configuração."""
        if hasattr(self, 'orchestrator') and self.orchestrator:
            st.subheader("⚙️ Configuração Utilizada")
            
            # Mostrar configuração de forma mais organizada
            config = self.orchestrator.config
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Projeto:**")
                st.json(config.get('project', {}))
                
                st.markdown("**Dados:**")
                st.json(config.get('data', {}))
                
                st.markdown("**Modelo:**")
                st.json(config.get('model', {}))
            
            with col2:
                st.markdown("**Pré-processamento:**")
                st.json(config.get('preprocessing', {}))
                
                st.markdown("**Treinamento:**")
                st.json(config.get('training', {}))
                
                st.markdown("**Avaliação:**")
                st.json(config.get('evaluation', {}))
            
            # Opção para download da configuração
            config_yaml = yaml.dump(config, default_flow_style=False)
            st.download_button(
                label="📥 Download Configuração (YAML)",
                data=config_yaml,
                file_name=f"config_{datetime.now().strftime('%Y%m%d_%H%M%S')}.yaml",
                mime="text/yaml"
            )
    
    def _render_logs_tab(self):
        """Renderiza a aba de logs."""
        st.subheader("📝 Logs de Execução")
        
        # Mostrar logs do resultado se disponível
        if hasattr(self, 'results') and 'status' in self.results:
            st.text(f"Status final: {self.results['status']}")
            
            if 'error' in self.results:
                st.error(f"Erro: {self.results['error']}")
            
            # Informações de execução
            if 'execution_time' in self.results:
                st.info(f"⏱️ Tempo de execução: {self.results['execution_time']:.2f} segundos")
            
            # Informações dos dados
            if 'data_shape' in self.results:
                shape = self.results['data_shape']
                st.info(f"📊 Dataset processado: {shape[0]} linhas × {shape[1]} colunas")
        
        # Link para logs detalhados
        log_files = list(Path("logs").glob("*.log")) if Path("logs").exists() else []
        if log_files:
            st.info(f"📁 Logs detalhados disponíveis em: {log_files[-1]}")


# Executar aplicação
if __name__ == "__main__":
    app = StreamlitMLApp()
    app.run()
        
        #