#!/usr/bin/env python3
"""Evaluation script for fraud detection models.

This script evaluates trained fraud detection models and generates
comprehensive performance reports.
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Dict, Any

import numpy as np
import pandas as pd
import joblib
from omegaconf import OmegaConf

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent / "src"))

from risk.evaluation import FraudDetectionEvaluator, FraudDetectionVisualizer

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    """Main evaluation function."""
    parser = argparse.ArgumentParser(description="Evaluate fraud detection models")
    parser.add_argument(
        "--models-dir", 
        type=str, 
        default="assets",
        help="Directory containing trained models"
    )
    parser.add_argument(
        "--output-dir", 
        type=str, 
        default="assets",
        help="Directory to save evaluation outputs"
    )
    parser.add_argument(
        "--config-dir", 
        type=str, 
        default="configs",
        help="Directory containing configuration files"
    )
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Load configurations
    config_dir = Path(args.config_dir)
    eval_config = OmegaConf.load(config_dir / "evaluation.yaml")
    
    logger.info("Starting fraud detection model evaluation")
    logger.info(f"Models directory: {args.models_dir}")
    logger.info(f"Output directory: {output_dir}")
    
    try:
        # Load test data
        test_data_path = Path(args.models_dir) / "features.csv"
        if not test_data_path.exists():
            logger.error(f"Test data not found at {test_data_path}")
            logger.info("Please run training first to generate test data")
            return
        
        df_features = pd.read_csv(test_data_path)
        logger.info(f"Loaded test data: {len(df_features)} samples")
        
        # Load feature engineer
        fe_path = Path(args.models_dir) / "feature_engineer.joblib"
        if not fe_path.exists():
            logger.error(f"Feature engineer not found at {fe_path}")
            return
        
        feature_engineer = joblib.load(fe_path)
        logger.info("Loaded feature engineer")
        
        # Prepare test data
        X_test, y_test = feature_engineer.prepare_features(df_features, fit_transformers=False)
        
        # Load models
        models_dir = Path(args.models_dir)
        models = {}
        
        model_files = list(models_dir.glob("*_model.joblib"))
        if not model_files:
            logger.error("No trained models found")
            return
        
        for model_file in model_files:
            model_name = model_file.stem.replace("_model", "")
            try:
                models[model_name] = joblib.load(model_file)
                logger.info(f"Loaded {model_name} model")
            except Exception as e:
                logger.error(f"Failed to load {model_name}: {str(e)}")
        
        if not models:
            logger.error("No models could be loaded")
            return
        
        # Evaluate models
        logger.info("Evaluating models")
        evaluator = FraudDetectionEvaluator(eval_config)
        
        for model_name, model in models.items():
            logger.info(f"Evaluating {model_name} model")
            evaluator.evaluate_model(model, X_test, y_test, model_name)
        
        # Generate evaluation report
        logger.info("Generating evaluation report")
        report = evaluator.generate_report(output_dir / "evaluation_report.txt")
        
        # Model comparison
        comparison_df = evaluator.compare_models()
        comparison_df.to_csv(output_dir / "model_comparison.csv", index=False)
        
        logger.info("Model comparison results:")
        logger.info(f"\n{comparison_df.to_string(index=False)}")
        
        # Generate visualizations
        if eval_config.get('visualization', {}).get('enabled', True):
            logger.info("Generating visualizations")
            visualizer = FraudDetectionVisualizer(eval_config)
            
            # ROC curves
            roc_fig = visualizer.plot_roc_curves(evaluator)
            roc_fig.write_html(output_dir / "roc_curves.html")
            
            # Precision-Recall curves
            pr_fig = visualizer.plot_precision_recall_curves(evaluator)
            pr_fig.write_html(output_dir / "precision_recall_curves.html")
            
            # Confusion matrices
            cm_fig = visualizer.plot_confusion_matrices(evaluator)
            cm_fig.write_html(output_dir / "confusion_matrices.html")
            
            # Feature importance
            fi_fig = visualizer.plot_feature_importance(evaluator)
            if fi_fig.data:  # Check if figure has data
                fi_fig.write_html(output_dir / "feature_importance.html")
            
            logger.info("Visualizations saved to assets/ directory")
        
        logger.info("Evaluation completed successfully!")
        logger.info(f"All outputs saved to {output_dir}")
        
        # Print summary
        print("\n" + "="*80)
        print("EVALUATION SUMMARY")
        print("="*80)
        print(f"Models evaluated: {len(models)}")
        print(f"Test samples: {len(X_test):,}")
        print(f"Fraud cases: {y_test.sum():,} ({y_test.mean():.1%})")
        print(f"Best model (ROC AUC): {comparison_df.iloc[0]['Model']} ({comparison_df.iloc[0]['ROC AUC']:.4f})")
        print("="*80)
        
    except Exception as e:
        logger.error(f"Evaluation failed with error: {str(e)}")
        raise


if __name__ == "__main__":
    main()
