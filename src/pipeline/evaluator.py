# src/pipeline/evaluator.py
import logging
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_curve, auc,
    roc_auc_score, precision_recall_curve
)

class ModelEvaluator:
    """
    Classe responsável pela avaliação de modelos de ML.
    """
    
    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger('model_evaluator')
        
    def calculate_metrics(self, y_true, y_pred, y_pred_proba=None):
        """
        Calcula métricas de avaliação.
        
        Args:
            y_true: Valores reais
            y_pred: Predições
            y_pred_proba: Probabilidades preditas (opcional)
            
        Returns:
            dict: Métricas calculadas
        """
        self.logger.info("Calculando métricas de avaliação")
        
        metrics = {}
        
        # Métricas básicas
        if 'accuracy' in self.config['evaluation']['metrics']:
            metrics['accuracy'] = accuracy_score(y_true, y_pred)
            
        if 'precision' in self.config['evaluation']['metrics']:
            metrics['precision'] = precision_score(y_true, y_pred, average='weighted')
            
        if 'recall' in self.config['evaluation']['metrics']:
            metrics['recall'] = recall_score(y_true, y_pred, average='weighted')
            
        if 'f1' in self.config['evaluation']['metrics']:
            metrics['f1'] = f1_score(y_true, y_pred, average='weighted')
        
        # Matriz de confusão
        if 'confusion_matrix' in self.config['evaluation']['metrics']:
            metrics['confusion_matrix'] = confusion_matrix(y_true, y_pred)
        
        # AUC-ROC se probabilidades estão disponíveis
        if y_pred_proba is not None:
            try:
                if len(np.unique(y_true)) == 2:  # Classificação binária
                    metrics['roc_auc'] = roc_auc_score(y_true, y_pred_proba[:, 1])
                else:  # Classificação multiclasse
                    metrics['roc_auc'] = roc_auc_score(y_true, y_pred_proba, multi_class='ovr')
            except Exception as e:
                self.logger.warning(f"Não foi possível calcular ROC AUC: {str(e)}")
        
        # Relatório de classificação
        metrics['classification_report'] = classification_report(y_true, y_pred, output_dict=True)
        
        self.logger.info(f"Métricas calculadas: {list(metrics.keys())}")
        return metrics
    
    def generate_plots(self, y_true, y_pred, y_pred_proba=None, model=None, feature_names=None):
        """
        Gera visualizações para avaliação.
        
        Args:
            y_true: Valores reais
            y_pred: Predições
            y_pred_proba: Probabilidades preditas (opcional)
            model: Modelo treinado (opcional)
            feature_names: Nomes das features (opcional)
            
        Returns:
            dict: Gráficos gerados
        """
        self.logger.info("Gerando visualizações")
        
        plots = {}
        
        # Matriz de confusão
        if 'confusion_matrix' in self.config['evaluation']['plots']:
            plots['confusion_matrix'] = self._plot_confusion_matrix(y_true, y_pred)
        
        # Curva ROC
        if 'roc_curve' in self.config['evaluation']['plots'] and y_pred_proba is not None:
            plots['roc_curve'] = self._plot_roc_curve(y_true, y_pred_proba)
        
        # Importância das features
        if 'feature_importance' in self.config['evaluation']['plots'] and model is not None:
            plots['feature_importance'] = self._plot_feature_importance(model, feature_names)
        
        # Curva Precision-Recall
        if 'precision_recall_curve' in self.config['evaluation']['plots'] and y_pred_proba is not None:
            plots['precision_recall_curve'] = self._plot_precision_recall_curve(y_true, y_pred_proba)
        
        # Distribuição de probabilidades
        if 'probability_distribution' in self.config['evaluation']['plots'] and y_pred_proba is not None:
            plots['probability_distribution'] = self._plot_probability_distribution(y_true, y_pred_proba)
        
        self.logger.info(f"Gráficos gerados: {list(plots.keys())}")
        return plots
    
    def _plot_confusion_matrix(self, y_true, y_pred):
        """Gera gráfico da matriz de confusão."""
        cm = confusion_matrix(y_true, y_pred)
        
        # Calcular percentuais
        cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
        
        # Criar labels com valores absolutos e percentuais
        text = []
        for i in range(len(cm)):
            text_row = []
            for j in range(len(cm[0])):
                text_row.append(f"{cm[i][j]}<br>({cm_percent[i][j]:.1f}%)")
            text.append(text_row)
        
        fig = px.imshow(
            cm,
            text_auto=False,
            aspect="auto",
            title="Matriz de Confusão",
            labels=dict(x="Predito", y="Real", color="Contagem"),
            color_continuous_scale='Blues'
        )
        
        # Adicionar texto personalizado
        for i in range(len(cm)):
            for j in range(len(cm[0])):
                fig.add_annotation(
                    x=j, y=i,
                    text=text[i][j],
                    showarrow=False,
                    font=dict(color="white" if cm[i][j] > cm.max() / 2 else "black")
                )
        
        fig.update_layout(
            width=500,
            height=500,
            xaxis_title="Classe Predita",
            yaxis_title="Classe Real"
        )
        
        return fig
    
    def _plot_roc_curve(self, y_true, y_pred_proba):
        """Gera curva ROC."""
        classes = np.unique(y_true)
        
        if len(classes) == 2:
            # Classificação binária
            fpr, tpr, _ = roc_curve(y_true, y_pred_proba[:, 1])
            roc_auc = auc(fpr, tpr)
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=fpr, y=tpr,
                mode='lines',
                name=f'ROC Curve (AUC = {roc_auc:.3f})',
                line=dict(color='blue', width=2)
            ))
            
            # Linha diagonal (classificador aleatório)
            fig.add_trace(go.Scatter(
                x=[0, 1], y=[0, 1],
                mode='lines',
                name='Classificador Aleatório',
                line=dict(color='red', dash='dash')
            ))
            
        else:
            # Classificação multiclasse
            fig = go.Figure()
            
            for i, class_label in enumerate(classes):
                # One-vs-Rest para cada classe
                y_binary = (y_true == class_label).astype(int)
                fpr, tpr, _ = roc_curve(y_binary, y_pred_proba[:, i])
                roc_auc = auc(fpr, tpr)
                
                fig.add_trace(go.Scatter(
                    x=fpr, y=tpr,
                    mode='lines',
                    name=f'Classe {class_label} (AUC = {roc_auc:.3f})',
                    line=dict(width=2)
                ))
            
            # Linha diagonal
            fig.add_trace(go.Scatter(
                x=[0, 1], y=[0, 1],
                mode='lines',
                name='Classificador Aleatório',
                line=dict(color='black', dash='dash')
            ))
        
        fig.update_layout(
            title='Curva ROC',
            xaxis_title='Taxa de Falsos Positivos',
            yaxis_title='Taxa de Verdadeiros Positivos',
            width=600,
            height=500,
            showlegend=True
        )
        
        return fig
    
    def _plot_feature_importance(self, model, feature_names):
        """Gera gráfico de importância das features."""
        if not hasattr(model, 'feature_importances_'):
            return None
        
        importances = model.feature_importances_
        
        if feature_names is None:
            feature_names = [f'Feature {i}' for i in range(len(importances))]
        
        # Criar DataFrame para facilitar ordenação
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importances
        }).sort_values('importance', ascending=True)
        
        fig = px.bar(
            importance_df,
            x='importance',
            y='feature',
            orientation='h',
            title='Importância das Features',
            labels={'importance': 'Importância', 'feature': 'Feature'}
        )
        
        fig.update_layout(
            width=700,
            height=max(400, len(feature_names) * 25),
            showlegend=False
        )
        
        return fig
    
    def _plot_precision_recall_curve(self, y_true, y_pred_proba):
        """Gera curva Precision-Recall."""
        classes = np.unique(y_true)
        
        if len(classes) == 2:
            # Classificação binária
            precision, recall, _ = precision_recall_curve(y_true, y_pred_proba[:, 1])
            pr_auc = auc(recall, precision)
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=recall, y=precision,
                mode='lines',
                name=f'PR Curve (AUC = {pr_auc:.3f})',
                line=dict(color='blue', width=2)
            ))
            
        else:
            # Classificação multiclasse
            fig = go.Figure()
            
            for i, class_label in enumerate(classes):
                y_binary = (y_true == class_label).astype(int)
                precision, recall, _ = precision_recall_curve(y_binary, y_pred_proba[:, i])
                pr_auc = auc(recall, precision)
                
                fig.add_trace(go.Scatter(
                    x=recall, y=precision,
                    mode='lines',
                    name=f'Classe {class_label} (AUC = {pr_auc:.3f})',
                    line=dict(width=2)
                ))
        
        fig.update_layout(
            title='Curva Precision-Recall',
            xaxis_title='Recall',
            yaxis_title='Precision',
            width=600,
            height=500,
            showlegend=True
        )
        
        return fig
    
    def _plot_probability_distribution(self, y_true, y_pred_proba):
        """Gera distribuição de probabilidades por classe."""
        classes = np.unique(y_true)
        
        fig = make_subplots(
            rows=1, cols=len(classes),
            subplot_titles=[f'Classe {c}' for c in classes],
            shared_yaxis=True
        )
        
        for i, class_label in enumerate(classes):
            # Probabilidades para a classe atual
            proba_class = y_pred_proba[:, i]
            
            # Separar por classe real
            correct_pred = proba_class[y_true == class_label]
            incorrect_pred = proba_class[y_true != class_label]
            
            # Adicionar histogramas
            fig.add_trace(
                go.Histogram(
                    x=correct_pred,
                    name=f'Correto - Classe {class_label}',
                    opacity=0.7,
                    nbinsx=20
                ),
                row=1, col=i+1
            )
            
            fig.add_trace(
                go.Histogram(
                    x=incorrect_pred,
                    name=f'Incorreto - Classe {class_label}',
                    opacity=0.7,
                    nbinsx=20
                ),
                row=1, col=i+1
            )
        
        fig.update_layout(
            title='Distribuição de Probabilidades por Classe',
            width=300 * len(classes),
            height=400,
            showlegend=True
        )
        
        return fig
    
    def generate_evaluation_report(self, metrics, plots):
        """
        Gera relatório completo de avaliação.
        
        Args:
            metrics (dict): Métricas calculadas
            plots (dict): Gráficos gerados
            
        Returns:
            dict: Relatório estruturado
        """
        report = {
            'summary': {
                'accuracy': metrics.get('accuracy', 'N/A'),
                'f1_score': metrics.get('f1', 'N/A'),
                'precision': metrics.get('precision', 'N/A'),
                'recall': metrics.get('recall', 'N/A')
            },
            'detailed_metrics': metrics,
            'visualizations': plots,
            'recommendations': self._generate_recommendations(metrics)
        }
        
        return report
    
    def _generate_recommendations(self, metrics):
        """
        Gera recomendações baseadas nas métricas.
        
        Args:
            metrics (dict): Métricas calculadas
            
        Returns:
            list: Lista de recomendações
        """
        recommendations = []
        
        accuracy = metrics.get('accuracy', 0)
        precision = metrics.get('precision', 0)
        recall = metrics.get('recall', 0)
        f1 = metrics.get('f1', 0)
        
        # Recomendações baseadas na performance
        if accuracy < 0.7:
            recommendations.append("⚠️ Acurácia baixa (<70%). Considere: mais dados, feature engineering ou algoritmo diferente.")
        
        if precision < 0.7:
            recommendations.append("⚠️ Precisão baixa. O modelo tem muitos falsos positivos. Ajuste o threshold ou melhore features.")
        
        if recall < 0.7:
            recommendations.append("⚠️ Recall baixo. O modelo está perdendo muitos casos positivos. Considere balanceamento de classes.")
        
        if f1 < 0.7:
            recommendations.append("⚠️ F1-Score baixo. Há desbalanceamento entre precisão e recall.")
        
        # Recomendações específicas sobre desbalanceamento
        if abs(precision - recall) > 0.15:
            if precision > recall:
                recommendations.append("📊 Modelo conservador (alta precisão, baixo recall). Bom para minimizar falsos positivos.")
            else:
                recommendations.append("📊 Modelo liberal (baixa precisão, alto recall). Bom para capturar mais casos positivos.")
        
        # Recomendações sobre overfitting (se tivermos CV scores)
        if hasattr(self, 'cv_scores') and self.cv_scores:
            cv_std = self.cv_scores.get('std', 0)
            if cv_std > 0.1:
                recommendations.append("⚠️ Alta variabilidade no CV. Possível overfitting. Considere regularização ou mais dados.")
        
        # Recomendações positivas
        if accuracy >= 0.9:
            recommendations.append("✅ Excelente acurácia! Modelo performando muito bem.")
        elif accuracy >= 0.8:
            recommendations.append("✅ Boa acurácia. Modelo com performance sólida.")
        
        if not recommendations:
            recommendations.append("✅ Performance geral adequada. Monitore em produção.")
        
        return recommendations