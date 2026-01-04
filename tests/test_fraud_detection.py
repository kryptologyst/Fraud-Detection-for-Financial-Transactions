"""Tests for fraud detection package."""

import pytest
import numpy as np
import pandas as pd
from omegaconf import DictConfig

# Import modules to test
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from data.generator import TransactionDataGenerator
from features.engineering import FeatureEngineer
from models.fraud_detection import RandomForestFraudModel, XGBoostFraudModel


class TestDataGenerator:
    """Test data generation functionality."""
    
    def test_data_generator_initialization(self):
        """Test data generator initialization."""
        config = DictConfig({
            'seed': 42,
            'n_users': 100,
            'amount': {'log_mean': 4.5, 'log_std': 1.2},
            'merchant_categories': ['Grocery', 'Gas Station'],
            'merchant_probabilities': [0.7, 0.3],
            'locations': ['New York', 'Los Angeles'],
            'location_probabilities': [0.6, 0.4],
            'device_types': ['Mobile', 'Desktop'],
            'device_probabilities': [0.8, 0.2]
        })
        
        generator = TransactionDataGenerator(config)
        assert generator.config == config
        assert generator.rng is not None
    
    def test_generate_transactions(self):
        """Test transaction generation."""
        config = DictConfig({
            'seed': 42,
            'n_users': 100,
            'amount': {'log_mean': 4.5, 'log_std': 1.2},
            'merchant_categories': ['Grocery', 'Gas Station'],
            'merchant_probabilities': [0.7, 0.3],
            'locations': ['New York', 'Los Angeles'],
            'location_probabilities': [0.6, 0.4],
            'device_types': ['Mobile', 'Desktop'],
            'device_probabilities': [0.8, 0.2]
        })
        
        generator = TransactionDataGenerator(config)
        df = generator.generate_transactions(n_samples=100, fraud_rate=0.1)
        
        assert len(df) == 100
        assert 'is_fraud' in df.columns
        assert 'amount' in df.columns
        assert 'user_id' in df.columns
        assert df['is_fraud'].sum() > 0  # Should have some fraud cases


class TestFeatureEngineer:
    """Test feature engineering functionality."""
    
    def test_feature_engineer_initialization(self):
        """Test feature engineer initialization."""
        config = {'feature_selection': {'enabled': False}}
        engineer = FeatureEngineer(config)
        assert engineer.config == config
    
    def test_create_features(self):
        """Test feature creation."""
        # Create sample data
        data = {
            'transaction_id': [1, 2, 3],
            'amount': [100, 200, 50],
            'user_id': [1, 2, 1],
            'timestamp': ['2024-01-01 10:00:00', '2024-01-01 15:00:00', '2024-01-01 20:00:00'],
            'merchant_category': ['Grocery', 'Gas Station', 'Restaurant'],
            'location': ['New York', 'Los Angeles', 'Chicago'],
            'device_type': ['Mobile', 'Desktop', 'Mobile']
        }
        
        df = pd.DataFrame(data)
        config = {'feature_selection': {'enabled': False}}
        engineer = FeatureEngineer(config)
        
        df_features = engineer.create_features(df)
        
        assert len(df_features) == 3
        assert len(df_features.columns) > len(df.columns)  # Should have more features


class TestModels:
    """Test model functionality."""
    
    def test_random_forest_model(self):
        """Test Random Forest model."""
        config = {
            'n_estimators': 10,
            'max_depth': 5,
            'random_state': 42
        }
        
        model = RandomForestFraudModel(config)
        assert model.model is not None
        assert not model.is_trained
    
    def test_xgboost_model(self):
        """Test XGBoost model."""
        config = {
            'n_estimators': 10,
            'max_depth': 3,
            'random_state': 42
        }
        
        model = XGBoostFraudModel(config)
        assert model.model is not None
        assert not model.is_trained


if __name__ == "__main__":
    pytest.main([__file__])
