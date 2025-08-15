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
        
    def run(self):
        """Roda a aplicação Streamlit."""
        self._render_header()
        self._render_sidebar()
        self._render_main_content()
    
    def _render_header(self):
        """Renderiza o cabeçalho da aplicação."""
        st.title("📈 DataVision EBAC SEMANTIX")
        st.markdown("---")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Status", "Pronto", delta=None)
        with col2:
            st.metric("Modelos Disponíveis", "3", delta="+1")
        with col3:
            st.metric("Pipelines Executados", len(st.session_state.get('executions', [])))
    
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
                df = pd.read_feather("data/raw/credit_scoring.ftr")
                return df, "Credit Scoring Dataset"
                
            elif data_source == "Hipertension":
                df = pd.read_csv("data/raw/hypertension_dataset.csv")
                return df, "Hypertension Dataset"
                
            elif data_source == "Phone addiction":
                df = pd.read_csv("data/raw/teen_phone_addiction_dataset.csv")
                return df, "Teen Phone Addiction Dataset"
                
        except Exception as e:
            st.error(f"Erro ao carregar dataset: {str(e)}")
            return None, None
            
        return None, None
    
    def _render_sidebar(self):
        """Renderiza a barra lateral com configurações."""
        st.sidebar.title("⚙️ Configurações do Pipeline")
        
        # Seção 1: Fonte dos dados
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
        
        # Recuperar dados da sessão se existirem
        self.current_data = st.session_state.get('current_data')
        
        # Seção 2: Configuração do modelo
        st.sidebar.subheader("🤖 Modelo")
        algorithm = st.sidebar.selectbox(
            "Algoritmo:",
            ["random_forest", "logistic_regression", "svm"],
            help="Escolha o algoritmo de ML"
        )
        
        # Parâmetros específicos do algoritmo
        self._render_algorithm_params(algorithm)
        
        # Seção 3: Configurações de avaliação
        st.sidebar.subheader("📊 Avaliação")
        test_size = st.sidebar.slider("Tamanho do teste:", 0.1, 0.5, 0.2, 0.05)
        cv_folds = st.sidebar.slider("Cross-validation:", 3, 10, 5)
        
        metrics = st.sidebar.multiselect(
            "Métricas:",
            ["accuracy", "precision", "recall", "f1"],
            default=["accuracy", "f1"]
        )
        
        # Seção 4: Configuração de pipeline
        st.sidebar.subheader("🔧 Pipeline")
        steps = st.sidebar.multiselect(
            "Etapas:",
            ["load_data", "preprocess_data", "train_model", "evaluate_model", "save_results"],
            default=["load_data", "preprocess_data", "train_model", "evaluate_model"]
        )
        
        # Botão para executar (só aparece se houver dados carregados)
        if self.current_data is not None:
            if st.sidebar.button("🚀 Executar Pipeline", type="primary"):
                self._execute_pipeline(
                    data_source, uploaded_file, algorithm, 
                    test_size, cv_folds, metrics, steps
                )
        else:
            st.sidebar.info("👆 Selecione um dataset para continuar")
        
        # Seção 5: Configurações avançadas
        with st.sidebar.expander("⚙️ Configurações Avançadas"):
            scaling = st.selectbox("Normalização:", ["standard", "minmax", "robust", "none"])
            random_state = st.number_input("Random State:", value=42, min_value=0)
    
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
        
        # 1. Converter colunas de datetime para features numéricas
        datetime_cols = df_processed.select_dtypes(include=['datetime64', 'datetime']).columns
        for col in datetime_cols:
            # Extrair features de data
            df_processed[f'{col}_year'] = pd.to_datetime(df_processed[col]).dt.year
            df_processed[f'{col}_month'] = pd.to_datetime(df_processed[col]).dt.month
            df_processed[f'{col}_day'] = pd.to_datetime(df_processed[col]).dt.day
            df_processed[f'{col}_dayofweek'] = pd.to_datetime(df_processed[col]).dt.dayofweek
            # Remover coluna original
            df_processed.drop(col, axis=1, inplace=True)
            st.info(f"✅ Convertida coluna de data '{col}' em features numéricas")
        
        # 2. Detectar e converter colunas que podem ser Timestamp como objeto
        for col in df_processed.columns:
            if df_processed[col].dtype == 'object':
                # Tentar converter para datetime
                try:
                    pd.to_datetime(df_processed[col])
                    # Se conseguiu converter, é uma coluna de data
                    df_processed[f'{col}_year'] = pd.to_datetime(df_processed[col]).dt.year
                    df_processed[f'{col}_month'] = pd.to_datetime(df_processed[col]).dt.month
                    df_processed[f'{col}_day'] = pd.to_datetime(df_processed[col]).dt.day
                    df_processed[f'{col}_dayofweek'] = pd.to_datetime(df_processed[col]).dt.dayofweek
                    df_processed.drop(col, axis=1, inplace=True)
                    st.info(f"✅ Detectada e convertida coluna de data '{col}' em features numéricas")
                except:
                    # Se não conseguiu, verificar se é categórica
                    if df_processed[col].nunique() < 50:  # Assumir categórica se < 50 valores únicos
                        # Encoding de variáveis categóricas
                        df_processed[col] = pd.Categorical(df_processed[col]).codes
                        st.info(f"✅ Convertida variável categórica '{col}' para códigos numéricos")
        
        # 3. Garantir que todas as colunas sejam numéricas (exceto target se existir)
        non_numeric_cols = []
        for col in df_processed.columns:
            if col != 'target' and not pd.api.types.is_numeric_dtype(df_processed[col]):
                non_numeric_cols.append(col)
        
        if non_numeric_cols:
            st.warning(f"⚠️ Removendo colunas não numéricas: {non_numeric_cols}")
            df_processed.drop(non_numeric_cols, axis=1, inplace=True)
        
        # 4. Tratar valores infinitos
        df_processed.replace([np.inf, -np.inf], np.nan, inplace=True)
        
        # 5. Verificar se existe coluna target
        if 'target' not in df_processed.columns:
            # Tentar detectar coluna target baseado em nomes comuns
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
                st.info(f"✅ Coluna '{target_col}' definida como target")
            else:
                # Se não encontrou, usar a última coluna
                last_col = df_processed.columns[-1]
                df_processed['target'] = df_processed[last_col]
                df_processed.drop(last_col, axis=1, inplace=True)
                st.warning(f"⚠️ Usando última coluna '{last_col}' como target")
        
        return df_processed
    def _execute_pipeline(self, data_source, uploaded_file, algorithm, test_size, cv_folds, metrics, steps):
        """Executa o pipeline com as configurações especificadas."""
        try:
            with st.spinner("🔄 Executando pipeline..."):
                # Criar configuração dinâmica
                config = self._create_config(
                    data_source, algorithm, test_size, cv_folds, metrics, steps
                )
                
                # Determinar que dados usar e pré-processar
                if data_source == "upload" and uploaded_file is not None:
                    # Para upload, usar os dados carregados
                    pipeline_data = self._preprocess_data_for_ml(self.current_data)
                elif data_source in ["Credit", "Hipertension", "Phone addiction"]:
                    # Para datasets personalizados, usar os dados já carregados
                    pipeline_data = self._preprocess_data_for_ml(self.current_data)
                    # Modificar config para usar dados externos
                    config['data']['source'] = 'external'
                else:
                    # Para datasets sklearn, deixar o orquestrador carregar
                    pipeline_data = None
                
                # Mostrar info sobre pré-processamento
                if pipeline_data is not None:
                    st.info(f"📊 Dados pré-processados: {pipeline_data.shape[0]} linhas, {pipeline_data.shape[1]} colunas")
                
                # Inicializar e executar orquestrador
                self.orchestrator = MLOrchestrator(config_dict=config)
                self.results = self.orchestrator.run_pipeline(pipeline_data)
                
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
            
            # Debug: mostrar mais detalhes do erro
            with st.expander("🔍 Detalhes do Erro (Debug)"):
                st.write(f"**Data Source:** {data_source}")
                st.write(f"**Config:** {config}")
                st.write(f"**Dados carregados:** {self.current_data is not None}")
                if self.current_data is not None:
                    st.write(f"**Shape dos dados originais:** {self.current_data.shape}")
                    st.write(f"**Tipos de dados originais:**")
                    st.write(self.current_data.dtypes)
                    
                    # Mostrar dados após pré-processamento
                    try:
                        processed_data = self._preprocess_data_for_ml(self.current_data)
                        if processed_data is not None:
                            st.write(f"**Shape após pré-processamento:** {processed_data.shape}")
                            st.write(f"**Tipos após pré-processamento:**")
                            st.write(processed_data.dtypes)
                    except Exception as prep_error:
                        st.write(f"**Erro no pré-processamento:** {prep_error}")
                
                st.write(f"**Erro completo:** {str(e)}")
                
                # Mostrar traceback se possível
                import traceback
                st.code(traceback.format_exc())
    
    def _create_config(self, data_source, algorithm, test_size, cv_folds, metrics, steps):
        """Cria configuração baseada nos inputs da interface."""
        model_params = st.session_state.get('model_params', {})
        
        # Mapear nomes dos datasets personalizados para o orquestrador
        data_source_mapping = {
            'Credit': 'external',
            'Hipertension': 'external', 
            'Phone addiction': 'external'
        }
        
        # Usar mapeamento se necessário
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
            'pipeline_steps': steps
        }
        
        return config
    
    def _render_main_content(self):
        """Renderiza o conteúdo principal."""
        if hasattr(self, 'results') and self.results:
            self._render_results()
        elif self.current_data is not None:
            self._render_data_preview()
        else:
            self._render_welcome()
    
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
            st.dataframe(self.current_data.head(10), use_container_width=True)
            
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
        
        ### Como usar:
        1. **Selecione um dataset** na barra lateral
        2. **Visualize os dados** antes do processamento
        3. **Configure o algoritmo** e seus parâmetros
        4. **Defina as métricas** de avaliação
        5. **Execute o pipeline** e veja os resultados
        
        **Comece selecionando um dataset na barra lateral!**
        """)
        
        # Seção de Datasets Disponíveis
        st.markdown("---")
        st.subheader("Datasets Disponíveis")
        st.markdown("Conheça os datasets que você pode usar nesta aplicação:")
        
        # Datasets Clássicos (Sklearn)
        st.markdown("### Datasets Clássicos (Educacionais)")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("Iris Dataset", key="iris_info"):
                st.info("""
                **Iris Dataset**
                - **Problema**: Classificação de espécies de flores íris
                - **Classes**: 3 (Setosa, Versicolor, Virginica)  
                - **Features**: 4 (comprimento/largura de pétalas e sépalas)
                - **Amostras**: 150 (50 por classe)
                - **Uso**: Perfeito para iniciantes - classificação multiclasse simples
                - **Origem**: Ronald Fisher (1936)
                """)
        
        with col2:
            if st.button("Wine Dataset", key="wine_info"):
                st.info("""
                **Wine Dataset**
                - **Problema**: Classificação de vinhos por origem
                - **Classes**: 3 (diferentes cultivares)
                - **Features**: 13 (análises químicas: álcool, ácido málico, etc.)
                - **Amostras**: 178 vinhos
                - **Uso**: Classificação com mais complexidade
                - **Origem**: Vinhos da região de Piemonte, Itália
                """)
        
        with col3:
            if st.button("Breast Cancer", key="cancer_info"):
                st.info("""
                **Breast Cancer Dataset**
                - **Problema**: Diagnóstico de câncer de mama
                - **Classes**: 2 (Maligno, Benigno)
                - **Features**: 30 (características dos núcleos celulares)
                - **Amostras**: 569 casos
                - **Uso**: Classificação binária - aplicação médica importante
                - **Origem**: Hospital da Universidade de Wisconsin
                """)
        
        # Datasets Personalizados
        st.markdown("### Datasets Personalizados (Projetos Reais)")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("Credit Scoring", key="credit_info"):
                st.info("""
                **Credit Scoring Dataset**
                - **Problema**: Análise de risco de crédito
                - **Objetivo**: Prever se um cliente vai pagar o empréstimo
                - **Tipo**: Classificação binária (Aprovado/Negado)
                - **Aplicação**: Bancos e fintechs
                - **Importância**: Decisões financeiras automatizadas
                - **Desafios**: Balanceamento, interpretabilidade
                """)
        
        with col2:
            if st.button("Hypertension", key="hypertension_info"):
                st.info("""
                **Hypertension Dataset**
                - **Problema**: Predição de hipertensão arterial
                - **Objetivo**: Identificar pacientes com risco de hipertensão
                - **Tipo**: Classificação médica
                - **Aplicação**: Diagnóstico preventivo
                - **Importância**: Saúde pública - prevenção de doenças cardiovasculares
                - **Features**: Dados demográficos, estilo de vida, exames
                """)
        
        with col3:
            if st.button("Phone Addiction", key="phone_info"):
                st.info("""
                **Teen Phone Addiction Dataset**
                - **Problema**: Identificação de vício em smartphones
                - **Objetivo**: Detectar adolescentes com uso problemático do celular
                - **Tipo**: Classificação comportamental
                - **Aplicação**: Saúde mental, bem-estar digital
                - **Importância**: Problema crescente na era digital
                - **Features**: Padrões de uso, comportamento, dados psicológicos
                """)
        
        # Recursos da aplicação
        st.markdown("---")
        st.markdown("### Recursos da Aplicação")
        
        feature_col1, feature_col2 = st.columns(2)
        
        with feature_col1:
            st.markdown("""
            **Análise de Dados:**
            - Prévia interativa dos datasets
            - Estatísticas descritivas automáticas
            - Visualizações exploratórias
            - Análise de qualidade dos dados
            - Detecção de valores faltantes
            - Análise de correlações
            """)
        
        with feature_col2:
            st.markdown("""
            **Machine Learning:**
            - Múltiplos algoritmos (RF, SVM, LogReg)
            - Configuração de hiperparâmetros
            - Cross-validation automática
            - Métricas de avaliação completas
            - Visualizações de resultados
            - Histórico de experimentos
            """)
        
        # Histórico de execuções
        if 'executions' in st.session_state and st.session_state.executions:
            st.markdown("---")
            st.subheader("Histórico de Execuções")
            
            for i, execution in enumerate(reversed(st.session_state.executions[-5:])):
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
    
    def _render_results(self):
        """Renderiza os resultados do pipeline."""
        st.header("📊 Resultados do Pipeline")
        
        # Status e métricas gerais
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            status = self.results.get('status', 'unknown')
            st.metric("Status", status.title(), delta=None)
            
        with col2:
            exec_time = self.results.get('execution_time', 0)
            st.metric("Tempo (s)", f"{exec_time:.2f}")
            
        with col3:
            if 'data_shape' in self.results:
                shape = self.results['data_shape']
                st.metric("Amostras", shape[0])
                
        with col4:
            if 'data_shape' in self.results:
                shape = self.results['data_shape']
                st.metric("Features", shape[1])
        
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
    
    def _render_metrics_tab(self):
        """Renderiza a aba de métricas."""
        if 'evaluation' in self.results:
            metrics = self.results['evaluation']['metrics']
            
            st.subheader("🎯 Métricas de Performance")
            
            # Métricas principais em colunas
            cols = st.columns(len([k for k, v in metrics.items() if isinstance(v, (int, float))]))
            
            col_idx = 0
            for metric, value in metrics.items():
                if isinstance(value, (int, float)):
                    with cols[col_idx]:
                        st.metric(metric.replace('_', ' ').title(), f"{value:.4f}")
                    col_idx += 1
            
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
            st.json(self.orchestrator.config)
    
    def _render_logs_tab(self):
        """Renderiza a aba de logs."""
        st.subheader("📝 Logs de Execução")
        
        # Aqui você pode mostrar logs do arquivo ou do logger
        st.info("Logs detalhados estão disponíveis no arquivo logs/app.log")
        
        if hasattr(self, 'results') and 'status' in self.results:
            st.text(f"Status final: {self.results['status']}")
            
            if 'error' in self.results:
                st.error(f"Erro: {self.results['error']}")

# Executar aplicação
if __name__ == "__main__":
    app = StreamlitMLApp()
    app.run()