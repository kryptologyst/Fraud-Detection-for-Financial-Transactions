"""Fraud Detection Research Package.

This package provides comprehensive fraud detection capabilities for
financial transactions using advanced machine learning techniques.

For research and educational purposes only.
"""

__version__ = "1.0.0"
__author__ = "Research Team"
__email__ = "research@example.com"

# Import main modules
from .data.generator import TransactionDataGenerator
from .features.engineering import FeatureEngineer
from .models.fraud_detection import (
    FraudDetectionModel,
    RandomForestFraudModel,
    XGBoostFraudModel,
    LightGBMFraudModel,
    IsolationForestFraudModel,
    NeuralNetworkFraudModel,
    EnsembleFraudModel,
    ModelTrainer
)
from .risk.evaluation import FraudDetectionEvaluator, FraudDetectionVisualizer

__all__ = [
    "TransactionDataGenerator",
    "FeatureEngineer",
    "FraudDetectionModel",
    "RandomForestFraudModel",
    "XGBoostFraudModel",
    "LightGBMFraudModel",
    "IsolationForestFraudModel",
    "NeuralNetworkFraudModel",
    "EnsembleFraudModel",
    "ModelTrainer",
    "FraudDetectionEvaluator",
    "FraudDetectionVisualizer",
]
