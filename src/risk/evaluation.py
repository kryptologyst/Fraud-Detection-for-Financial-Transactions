"""Evaluation metrics and analysis for fraud detection models.

This module provides comprehensive evaluation capabilities specifically
designed for fraud detection tasks.
"""

import logging
from typing import Dict, List, Optional, Tuple, Union, Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score, 
    precision_recall_curve, roc_curve, average_precision_score,
    precision_score, recall_score, f1_score, accuracy_score
)
from sklearn.model_selection import cross_val_score, StratifiedKFold
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import shap

logger = logging.getLogger(__name__)


class FraudDetectionEvaluator:
    """Comprehensive evaluator for fraud detection models."""
    
    def __init__(self, config: Dict[str, Any]) -> None:
        """Initialize the evaluator.
        
        Args:
            config: Evaluation configuration
        """
        self.config = config
        self.results = {}
        
    def evaluate_model(
        self, 
        model: Any, 
        X_test: pd.DataFrame, 
        y_test: pd.Series,
        model_name: str = "model"
    ) -> Dict[str, Any]:
        """Comprehensive evaluation of a fraud detection model.
        
        Args:
            model: Trained fraud detection model
            X_test: Test features
            y_test: Test labels
            model_name: Name of the model for reporting
            
        Returns:
            Dictionary containing evaluation results
        """
        logger.info(f"Evaluating {model_name} model")
        
        # Get predictions
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)
        
        # Handle binary classification probabilities
        if y_pred_proba.shape[1] == 2:
            fraud_proba = y_pred_proba[:, 1]
        else:
            fraud_proba = y_pred_proba.flatten()
            
        # Calculate metrics
        results = {
            'model_name': model_name,
            'predictions': y_pred,
            'probabilities': fraud_proba,
            'true_labels': y_test.values,
        }
        
        # Basic classification metrics
        results.update(self._calculate_classification_metrics(y_test, y_pred))
        
        # Fraud-specific metrics
        results.update(self._calculate_fraud_metrics(y_test, fraud_proba))
        
        # Cross-validation metrics
        if self.config.get('cross_validation', {}).get('enabled', True):
            cv_results = self._cross_validate_model(model, X_test, y_test)
            results['cross_validation'] = cv_results
            
        # Feature importance
        if hasattr(model, 'get_feature_importance'):
            feature_importance = model.get_feature_importance()
            if feature_importance is not None:
                results['feature_importance'] = feature_importance
                
        # Store results
        self.results[model_name] = results
        
        logger.info(f"Evaluation completed for {model_name}")
        return results
        
    def _calculate_classification_metrics(
        self, 
        y_true: pd.Series, 
        y_pred: np.ndarray
    ) -> Dict[str, float]:
        """Calculate standard classification metrics.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            
        Returns:
            Dictionary of classification metrics
        """
        return {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, zero_division=0),
            'recall': recall_score(y_true, y_pred, zero_division=0),
            'f1_score': f1_score(y_true, y_pred, zero_division=0),
            'specificity': self._calculate_specificity(y_true, y_pred),
        }
        
    def _calculate_fraud_metrics(
        self, 
        y_true: pd.Series, 
        y_pred_proba: np.ndarray
    ) -> Dict[str, float]:
        """Calculate fraud-specific metrics.
        
        Args:
            y_true: True labels
            y_pred_proba: Prediction probabilities
            
        Returns:
            Dictionary of fraud-specific metrics
        """
        # ROC AUC
        roc_auc = roc_auc_score(y_true, y_pred_proba)
        
        # Precision-Recall AUC
        pr_auc = average_precision_score(y_true, y_pred_proba)
        
        # Precision@K metrics
        precision_at_k = self._calculate_precision_at_k(y_true, y_pred_proba)
        
        # KS Statistic
        ks_statistic = self._calculate_ks_statistic(y_true, y_pred_proba)
        
        # Gini Coefficient
        gini_coefficient = 2 * roc_auc - 1
        
        return {
            'roc_auc': roc_auc,
            'pr_auc': pr_auc,
            'gini_coefficient': gini_coefficient,
            'ks_statistic': ks_statistic,
            'precision_at_10': precision_at_k.get(10, 0),
            'precision_at_50': precision_at_k.get(50, 0),
            'precision_at_100': precision_at_k.get(100, 0),
        }
        
    def _calculate_specificity(self, y_true: pd.Series, y_pred: np.ndarray) -> float:
        """Calculate specificity (True Negative Rate).
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            
        Returns:
            Specificity score
        """
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        return tn / (tn + fp) if (tn + fp) > 0 else 0
        
    def _calculate_precision_at_k(
        self, 
        y_true: pd.Series, 
        y_pred_proba: np.ndarray,
        k_values: List[int] = [10, 50, 100]
    ) -> Dict[int, float]:
        """Calculate precision at top K predictions.
        
        Args:
            y_true: True labels
            y_pred_proba: Prediction probabilities
            k_values: List of K values to calculate precision for
            
        Returns:
            Dictionary of precision@K scores
        """
        # Sort by probability (descending)
        sorted_indices = np.argsort(y_pred_proba)[::-1]
        sorted_labels = y_true.iloc[sorted_indices]
        
        precision_at_k = {}
        for k in k_values:
            if k <= len(sorted_labels):
                top_k_labels = sorted_labels[:k]
                precision_at_k[k] = top_k_labels.sum() / k
            else:
                precision_at_k[k] = 0
                
        return precision_at_k
        
    def _calculate_ks_statistic(self, y_true: pd.Series, y_pred_proba: np.ndarray) -> float:
        """Calculate Kolmogorov-Smirnov statistic.
        
        Args:
            y_true: True labels
            y_pred_proba: Prediction probabilities
            
        Returns:
            KS statistic
        """
        # Separate probabilities by class
        fraud_proba = y_pred_proba[y_true == 1]
        normal_proba = y_pred_proba[y_true == 0]
        
        # Calculate cumulative distributions
        fraud_cdf = np.sort(fraud_proba)
        normal_cdf = np.sort(normal_proba)
        
        # Calculate KS statistic
        ks_statistic = 0
        for threshold in np.linspace(0, 1, 100):
            fraud_rate = np.mean(fraud_proba >= threshold)
            normal_rate = np.mean(normal_proba >= threshold)
            ks_statistic = max(ks_statistic, abs(fraud_rate - normal_rate))
            
        return ks_statistic
        
    def _cross_validate_model(
        self, 
        model: Any, 
        X: pd.DataFrame, 
        y: pd.Series
    ) -> Dict[str, float]:
        """Perform cross-validation evaluation.
        
        Args:
            model: Model to evaluate
            X: Feature matrix
            y: Target labels
            
        Returns:
            Cross-validation results
        """
        cv_config = self.config.get('cross_validation', {})
        cv_folds = cv_config.get('folds', 5)
        
        # Stratified K-Fold for imbalanced data
        skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
        
        # Cross-validation scores
        cv_scores = cross_val_score(
            model.model if hasattr(model, 'model') else model,
            X, y,
            cv=skf,
            scoring='roc_auc'
        )
        
        return {
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'cv_scores': cv_scores.tolist()
        }
        
    def compare_models(self) -> pd.DataFrame:
        """Compare performance of all evaluated models.
        
        Returns:
            DataFrame with model comparison results
        """
        if not self.results:
            logger.warning("No models have been evaluated yet")
            return pd.DataFrame()
            
        comparison_data = []
        
        for model_name, results in self.results.items():
            comparison_data.append({
                'Model': model_name,
                'ROC AUC': results.get('roc_auc', 0),
                'PR AUC': results.get('pr_auc', 0),
                'Precision': results.get('precision', 0),
                'Recall': results.get('recall', 0),
                'F1 Score': results.get('f1_score', 0),
                'KS Statistic': results.get('ks_statistic', 0),
                'Gini Coefficient': results.get('gini_coefficient', 0),
                'Precision@10': results.get('precision_at_10', 0),
                'Precision@50': results.get('precision_at_50', 0),
            })
            
        comparison_df = pd.DataFrame(comparison_data)
        comparison_df = comparison_df.sort_values('ROC AUC', ascending=False)
        
        return comparison_df
        
    def generate_report(self, output_path: Optional[str] = None) -> str:
        """Generate comprehensive evaluation report.
        
        Args:
            output_path: Optional path to save report
            
        Returns:
            Report text
        """
        report_lines = []
        report_lines.append("=" * 80)
        report_lines.append("FRAUD DETECTION MODEL EVALUATION REPORT")
        report_lines.append("=" * 80)
        report_lines.append("")
        
        # Model comparison
        comparison_df = self.compare_models()
        if not comparison_df.empty:
            report_lines.append("MODEL COMPARISON")
            report_lines.append("-" * 40)
            report_lines.append(comparison_df.to_string(index=False))
            report_lines.append("")
            
        # Individual model details
        for model_name, results in self.results.items():
            report_lines.append(f"DETAILED RESULTS: {model_name.upper()}")
            report_lines.append("-" * 40)
            
            # Classification metrics
            report_lines.append("Classification Metrics:")
            report_lines.append(f"  Accuracy: {results.get('accuracy', 0):.4f}")
            report_lines.append(f"  Precision: {results.get('precision', 0):.4f}")
            report_lines.append(f"  Recall: {results.get('recall', 0):.4f}")
            report_lines.append(f"  F1 Score: {results.get('f1_score', 0):.4f}")
            report_lines.append(f"  Specificity: {results.get('specificity', 0):.4f}")
            report_lines.append("")
            
            # Fraud-specific metrics
            report_lines.append("Fraud Detection Metrics:")
            report_lines.append(f"  ROC AUC: {results.get('roc_auc', 0):.4f}")
            report_lines.append(f"  PR AUC: {results.get('pr_auc', 0):.4f}")
            report_lines.append(f"  Gini Coefficient: {results.get('gini_coefficient', 0):.4f}")
            report_lines.append(f"  KS Statistic: {results.get('ks_statistic', 0):.4f}")
            report_lines.append(f"  Precision@10: {results.get('precision_at_10', 0):.4f}")
            report_lines.append(f"  Precision@50: {results.get('precision_at_50', 0):.4f}")
            report_lines.append("")
            
            # Cross-validation results
            if 'cross_validation' in results:
                cv_results = results['cross_validation']
                report_lines.append("Cross-Validation Results:")
                report_lines.append(f"  Mean ROC AUC: {cv_results.get('cv_mean', 0):.4f}")
                report_lines.append(f"  Std ROC AUC: {cv_results.get('cv_std', 0):.4f}")
                report_lines.append("")
                
        # Feature importance
        for model_name, results in self.results.items():
            if 'feature_importance' in results:
                feature_importance = results['feature_importance']
                report_lines.append(f"TOP FEATURES: {model_name.upper()}")
                report_lines.append("-" * 40)
                top_features = feature_importance.head(10)
                for feature, importance in top_features.items():
                    report_lines.append(f"  {feature}: {importance:.4f}")
                report_lines.append("")
                
        report_lines.append("=" * 80)
        report_lines.append("END OF REPORT")
        report_lines.append("=" * 80)
        
        report_text = "\n".join(report_lines)
        
        if output_path:
            with open(output_path, 'w') as f:
                f.write(report_text)
            logger.info(f"Report saved to {output_path}")
            
        return report_text


class FraudDetectionVisualizer:
    """Visualization tools for fraud detection results."""
    
    def __init__(self, config: Dict[str, Any]) -> None:
        """Initialize visualizer."""
        self.config = config
        
    def plot_roc_curves(self, evaluator: FraudDetectionEvaluator, output_path: Optional[str] = None) -> go.Figure:
        """Plot ROC curves for all models.
        
        Args:
            evaluator: Evaluator with results
            output_path: Optional path to save plot
            
        Returns:
            Plotly figure
        """
        fig = go.Figure()
        
        # Add diagonal line
        fig.add_trace(go.Scatter(
            x=[0, 1], y=[0, 1],
            mode='lines',
            line=dict(dash='dash', color='gray'),
            name='Random Classifier',
            showlegend=True
        ))
        
        # Add ROC curves for each model
        for model_name, results in evaluator.results.items():
            y_true = results['true_labels']
            y_pred_proba = results['probabilities']
            
            fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
            roc_auc = results.get('roc_auc', 0)
            
            fig.add_trace(go.Scatter(
                x=fpr, y=tpr,
                mode='lines',
                name=f'{model_name} (AUC = {roc_auc:.3f})',
                line=dict(width=2)
            ))
            
        fig.update_layout(
            title='ROC Curves Comparison',
            xaxis_title='False Positive Rate',
            yaxis_title='True Positive Rate',
            width=800,
            height=600
        )
        
        if output_path:
            fig.write_html(output_path)
            
        return fig
        
    def plot_precision_recall_curves(self, evaluator: FraudDetectionEvaluator, output_path: Optional[str] = None) -> go.Figure:
        """Plot Precision-Recall curves for all models.
        
        Args:
            evaluator: Evaluator with results
            output_path: Optional path to save plot
            
        Returns:
            Plotly figure
        """
        fig = go.Figure()
        
        # Add baseline (fraud rate)
        fraud_rate = np.mean([results['true_labels'].mean() for results in evaluator.results.values()])
        fig.add_trace(go.Scatter(
            x=[0, 1], y=[fraud_rate, fraud_rate],
            mode='lines',
            line=dict(dash='dash', color='gray'),
            name=f'Baseline (Fraud Rate = {fraud_rate:.3f})',
            showlegend=True
        ))
        
        # Add PR curves for each model
        for model_name, results in evaluator.results.items():
            y_true = results['true_labels']
            y_pred_proba = results['probabilities']
            
            precision, recall, _ = precision_recall_curve(y_true, y_pred_proba)
            pr_auc = results.get('pr_auc', 0)
            
            fig.add_trace(go.Scatter(
                x=recall, y=precision,
                mode='lines',
                name=f'{model_name} (AUC = {pr_auc:.3f})',
                line=dict(width=2)
            ))
            
        fig.update_layout(
            title='Precision-Recall Curves Comparison',
            xaxis_title='Recall',
            yaxis_title='Precision',
            width=800,
            height=600
        )
        
        if output_path:
            fig.write_html(output_path)
            
        return fig
        
    def plot_confusion_matrices(self, evaluator: FraudDetectionEvaluator, output_path: Optional[str] = None) -> go.Figure:
        """Plot confusion matrices for all models.
        
        Args:
            evaluator: Evaluator with results
            output_path: Optional path to save plot
            
        Returns:
            Plotly figure
        """
        n_models = len(evaluator.results)
        fig = make_subplots(
            rows=1, cols=n_models,
            subplot_titles=list(evaluator.results.keys()),
            specs=[[{'type': 'heatmap'} for _ in range(n_models)]]
        )
        
        for i, (model_name, results) in enumerate(evaluator.results.items()):
            y_true = results['true_labels']
            y_pred = results['predictions']
            
            cm = confusion_matrix(y_true, y_pred)
            
            fig.add_trace(
                go.Heatmap(
                    z=cm,
                    x=['Predicted Normal', 'Predicted Fraud'],
                    y=['Actual Normal', 'Actual Fraud'],
                    text=cm,
                    texttemplate='%{text}',
                    textfont={'size': 12},
                    colorscale='Blues',
                    showscale=False
                ),
                row=1, col=i+1
            )
            
        fig.update_layout(
            title='Confusion Matrices Comparison',
            width=300 * n_models,
            height=400
        )
        
        if output_path:
            fig.write_html(output_path)
            
        return fig
        
    def plot_feature_importance(self, evaluator: FraudDetectionEvaluator, output_path: Optional[str] = None) -> go.Figure:
        """Plot feature importance for models that support it.
        
        Args:
            evaluator: Evaluator with results
            output_path: Optional path to save plot
            
        Returns:
            Plotly figure
        """
        models_with_importance = {
            name: results for name, results in evaluator.results.items()
            if 'feature_importance' in results
        }
        
        if not models_with_importance:
            logger.warning("No models with feature importance found")
            return go.Figure()
            
        n_models = len(models_with_importance)
        fig = make_subplots(
            rows=1, cols=n_models,
            subplot_titles=list(models_with_importance.keys())
        )
        
        for i, (model_name, results) in enumerate(models_with_importance.items()):
            feature_importance = results['feature_importance'].head(10)
            
            fig.add_trace(
                go.Bar(
                    x=feature_importance.values,
                    y=feature_importance.index,
                    orientation='h',
                    name=model_name
                ),
                row=1, col=i+1
            )
            
        fig.update_layout(
            title='Feature Importance Comparison',
            width=300 * n_models,
            height=600
        )
        
        if output_path:
            fig.write_html(output_path)
            
        return fig
