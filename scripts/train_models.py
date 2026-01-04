#!/usr/bin/env python3
"""Training script for fraud detection models.

This script trains multiple fraud detection models and evaluates their performance.
For research and educational purposes only.
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Dict, Any

import numpy as np
import pandas as pd
from omegaconf import OmegaConf

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent / "src"))

from data.generator import TransactionDataGenerator, load_config, save_data
from features.engineering import FeatureEngineer
from models.fraud_detection import ModelTrainer
from risk.evaluation import FraudDetectionEvaluator, FraudDetectionVisualizer

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description="Train fraud detection models")
    parser.add_argument(
        "--config-dir", 
        type=str, 
        default="configs",
        help="Directory containing configuration files"
    )
    parser.add_argument(
        "--output-dir", 
        type=str, 
        default="assets",
        help="Directory to save outputs"
    )
    parser.add_argument(
        "--n-samples", 
        type=int, 
        default=10000,
        help="Number of samples to generate"
    )
    parser.add_argument(
        "--fraud-rate", 
        type=float, 
        default=0.05,
        help="Fraud rate in the dataset"
    )
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Load configurations
    config_dir = Path(args.config_dir)
    data_config = load_config(config_dir / "data.yaml")
    model_config = load_config(config_dir / "models.yaml")
    eval_config = load_config(config_dir / "evaluation.yaml")
    
    # Set random seeds for reproducibility
    np.random.seed(data_config.seed)
    
    logger.info("Starting fraud detection model training")
    logger.info(f"Configuration directory: {config_dir}")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Number of samples: {args.n_samples}")
    logger.info(f"Fraud rate: {args.fraud_rate:.1%}")
    
    try:
        # Step 1: Generate synthetic data
        logger.info("Step 1: Generating synthetic transaction data")
        data_generator = TransactionDataGenerator(data_config)
        df = data_generator.generate_transactions(
            n_samples=args.n_samples,
            fraud_rate=args.fraud_rate
        )
        
        # Save raw data
        save_data(df, output_dir / "raw_transactions.csv")
        logger.info(f"Generated {len(df)} transactions with {df['is_fraud'].sum()} fraud cases")
        
        # Step 2: Feature engineering
        logger.info("Step 2: Engineering features")
        feature_engineer = FeatureEngineer(eval_config)
        df_features = feature_engineer.create_features(df)
        
        # Save features
        save_data(df_features, output_dir / "features.csv")
        logger.info(f"Created {len(df_features.columns)} features")
        
        # Step 3: Prepare data for training
        logger.info("Step 3: Preparing data for training")
        X, y = feature_engineer.prepare_features(df_features, fit_transformers=True)
        
        # Train-test split
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=data_config.seed, stratify=y
        )
        
        logger.info(f"Training set: {len(X_train)} samples")
        logger.info(f"Test set: {len(X_test)} samples")
        logger.info(f"Training fraud rate: {y_train.mean():.1%}")
        logger.info(f"Test fraud rate: {y_test.mean():.1%}")
        
        # Step 4: Train models
        logger.info("Step 4: Training fraud detection models")
        trainer = ModelTrainer(model_config)
        trained_models = trainer.train_all_models(X_train, y_train)
        
        logger.info(f"Trained {len(trained_models)} models")
        
        # Step 5: Evaluate models
        logger.info("Step 5: Evaluating models")
        evaluator = FraudDetectionEvaluator(eval_config)
        
        for model_name, model in trained_models.items():
            logger.info(f"Evaluating {model_name} model")
            evaluator.evaluate_model(model, X_test, y_test, model_name)
        
        # Step 6: Generate evaluation report
        logger.info("Step 6: Generating evaluation report")
        report = evaluator.generate_report(output_dir / "evaluation_report.txt")
        
        # Model comparison
        comparison_df = evaluator.compare_models()
        comparison_df.to_csv(output_dir / "model_comparison.csv", index=False)
        
        logger.info("Model comparison results:")
        logger.info(f"\n{comparison_df.to_string(index=False)}")
        
        # Step 7: Generate visualizations
        if eval_config.get('visualization', {}).get('enabled', True):
            logger.info("Step 7: Generating visualizations")
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
        
        # Step 8: Save model artifacts
        logger.info("Step 8: Saving model artifacts")
        import joblib
        
        for model_name, model in trained_models.items():
            model_path = output_dir / f"{model_name}_model.joblib"
            joblib.dump(model, model_path)
            logger.info(f"Saved {model_name} model to {model_path}")
        
        # Save feature engineer
        fe_path = output_dir / "feature_engineer.joblib"
        joblib.dump(feature_engineer, fe_path)
        logger.info(f"Saved feature engineer to {fe_path}")
        
        logger.info("Training completed successfully!")
        logger.info(f"All outputs saved to {output_dir}")
        
        # Print summary
        print("\n" + "="*80)
        print("TRAINING SUMMARY")
        print("="*80)
        print(f"Dataset size: {len(df):,} transactions")
        print(f"Fraud cases: {df['is_fraud'].sum():,} ({df['is_fraud'].mean():.1%})")
        print(f"Features created: {len(df_features.columns)}")
        print(f"Models trained: {len(trained_models)}")
        print(f"Best model (ROC AUC): {comparison_df.iloc[0]['Model']} ({comparison_df.iloc[0]['ROC AUC']:.4f})")
        print("="*80)
        
    except Exception as e:
        logger.error(f"Training failed with error: {str(e)}")
        raise


if __name__ == "__main__":
    main()
