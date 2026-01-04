"""Feature engineering for fraud detection.

This module provides feature extraction and engineering capabilities
for fraud detection models.
"""

import logging
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_selection import SelectKBest, f_classif

logger = logging.getLogger(__name__)


class FeatureEngineer:
    """Feature engineering for fraud detection models.
    
    This class handles feature extraction, transformation, and selection
    for fraud detection models.
    """
    
    def __init__(self, config: Dict) -> None:
        """Initialize feature engineer with configuration.
        
        Args:
            config: Configuration dictionary for feature engineering
        """
        self.config = config
        self.scalers = {}
        self.encoders = {}
        self.feature_selectors = {}
        
    def create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create comprehensive feature set for fraud detection.
        
        Args:
            df: Raw transaction DataFrame
            
        Returns:
            DataFrame with engineered features
        """
        logger.info("Creating features for fraud detection")
        
        # Start with base features
        features_df = df.copy()
        
        # Add temporal features
        features_df = self._add_temporal_features(features_df)
        
        # Add amount-based features
        features_df = self._add_amount_features(features_df)
        
        # Add user behavior features
        features_df = self._add_user_features(features_df)
        
        # Add merchant features
        features_df = self._add_merchant_features(features_df)
        
        # Add location features
        features_df = self._add_location_features(features_df)
        
        # Add interaction features
        features_df = self._add_interaction_features(features_df)
        
        # Add statistical features
        features_df = self._add_statistical_features(features_df)
        
        logger.info(f"Created {len(features_df.columns)} features")
        
        return features_df
    
    def _add_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add temporal features based on transaction timing.
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with temporal features
        """
        df = df.copy()
        
        # Convert timestamp to datetime
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Basic time features
        df['hour'] = df['timestamp'].dt.hour
        df['day_of_week'] = df['timestamp'].dt.dayofweek
        df['day_of_month'] = df['timestamp'].dt.day
        df['month'] = df['timestamp'].dt.month
        df['quarter'] = df['timestamp'].dt.quarter
        
        # Cyclical encoding for time features
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
        df['day_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
        df['day_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
        
        # Business hours indicator
        df['is_business_hours'] = (
            (df['hour'] >= 9) & (df['hour'] <= 17) & 
            (df['day_of_week'] < 5)
        ).astype(int)
        
        # Weekend indicator
        df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
        
        # Night time indicator
        df['is_night'] = ((df['hour'] >= 22) | (df['hour'] <= 6)).astype(int)
        
        return df
    
    def _add_amount_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add amount-based features.
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with amount features
        """
        df = df.copy()
        
        # Log transformation
        df['amount_log'] = np.log1p(df['amount'])
        
        # Square root transformation
        df['amount_sqrt'] = np.sqrt(df['amount'])
        
        # Amount categories
        df['amount_category'] = pd.cut(
            df['amount'],
            bins=[0, 50, 200, 500, 1000, float('inf')],
            labels=['low', 'medium', 'high', 'very_high', 'extreme']
        )
        
        # Amount percentiles
        df['amount_percentile'] = df['amount'].rank(pct=True)
        
        return df
    
    def _add_user_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add user behavior features.
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with user features
        """
        df = df.copy()
        
        # User transaction statistics
        user_stats = df.groupby('user_id').agg({
            'amount': ['count', 'mean', 'std', 'min', 'max'],
            'timestamp': ['min', 'max']
        }).round(2)
        
        user_stats.columns = ['_'.join(col).strip() for col in user_stats.columns]
        user_stats = user_stats.reset_index()
        
        # Merge user statistics
        df = df.merge(user_stats, on='user_id', how='left')
        
        # User activity features
        df['user_transaction_count'] = df['amount_count']
        df['user_avg_amount'] = df['amount_mean']
        df['user_amount_std'] = df['amount_std'].fillna(0)
        df['user_amount_range'] = df['amount_max'] - df['amount_min']
        
        # User recency (days since last transaction)
        df['user_recency'] = (
            pd.to_datetime(df['timestamp']) - pd.to_datetime(df['timestamp_max'])
        ).dt.days
        
        # User frequency (transactions per day)
        user_lifetime = (
            pd.to_datetime(df['timestamp_max']) - pd.to_datetime(df['timestamp_min'])
        ).dt.days + 1
        df['user_frequency'] = df['user_transaction_count'] / user_lifetime
        
        return df
    
    def _add_merchant_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add merchant-related features.
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with merchant features
        """
        df = df.copy()
        
        # Merchant transaction statistics
        merchant_stats = df.groupby('merchant_category').agg({
            'amount': ['count', 'mean', 'std'],
            'is_fraud': 'mean'
        }).round(2)
        
        merchant_stats.columns = ['_'.join(col).strip() for col in merchant_stats.columns]
        merchant_stats = merchant_stats.reset_index()
        
        # Merge merchant statistics
        df = df.merge(merchant_stats, on='merchant_category', how='left')
        
        # Merchant risk features
        df['merchant_transaction_count'] = df['amount_count']
        df['merchant_avg_amount'] = df['amount_mean']
        df['merchant_fraud_rate'] = df['is_fraud_mean'].fillna(0)
        
        return df
    
    def _add_location_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add location-based features.
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with location features
        """
        df = df.copy()
        
        # Location transaction statistics
        location_stats = df.groupby('location').agg({
            'amount': ['count', 'mean'],
            'is_fraud': 'mean'
        }).round(2)
        
        location_stats.columns = ['_'.join(col).strip() for col in location_stats.columns]
        location_stats = location_stats.reset_index()
        
        # Merge location statistics
        df = df.merge(location_stats, on='location', how='left')
        
        # Location risk features
        df['location_transaction_count'] = df['amount_count']
        df['location_avg_amount'] = df['amount_mean']
        df['location_fraud_rate'] = df['is_fraud_mean'].fillna(0)
        
        return df
    
    def _add_interaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add interaction features between different variables.
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with interaction features
        """
        df = df.copy()
        
        # Amount and time interactions
        df['amount_hour_interaction'] = df['amount'] * df['hour']
        df['amount_day_interaction'] = df['amount'] * df['day_of_week']
        
        # User and merchant interactions
        df['user_merchant_interaction'] = (
            df['user_id'].astype(str) + '_' + df['merchant_category'].astype(str)
        )
        
        # Amount and location interactions
        df['amount_location_interaction'] = df['amount'] * df['location_fraud_rate']
        
        return df
    
    def _add_statistical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add statistical features for anomaly detection.
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with statistical features
        """
        df = df.copy()
        
        # Z-scores for amount
        df['amount_zscore'] = (df['amount'] - df['amount'].mean()) / df['amount'].std()
        
        # Z-scores for user amount
        df['user_amount_zscore'] = (
            (df['amount'] - df['user_avg_amount']) / 
            (df['user_amount_std'] + 1e-8)
        )
        
        # Percentile ranks
        df['amount_percentile_rank'] = df['amount'].rank(pct=True)
        df['user_amount_percentile_rank'] = df.groupby('user_id')['amount'].rank(pct=True)
        
        return df
    
    def prepare_features(
        self, 
        df: pd.DataFrame, 
        target_column: str = 'is_fraud',
        fit_transformers: bool = True
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """Prepare features for model training.
        
        Args:
            df: Input DataFrame with features
            target_column: Name of target column
            fit_transformers: Whether to fit transformers
            
        Returns:
            Tuple of (features_df, target_series)
        """
        logger.info("Preparing features for model training")
        
        # Separate features and target
        feature_columns = [col for col in df.columns if col != target_column]
        X = df[feature_columns].copy()
        y = df[target_column].copy()
        
        # Handle categorical variables
        categorical_columns = X.select_dtypes(include=['object', 'category']).columns
        
        for col in categorical_columns:
            if fit_transformers:
                encoder = LabelEncoder()
                X[col] = encoder.fit_transform(X[col].astype(str))
                self.encoders[col] = encoder
            else:
                if col in self.encoders:
                    X[col] = self.encoders[col].transform(X[col].astype(str))
        
        # Handle missing values
        X = X.fillna(X.median())
        
        # Feature scaling
        if fit_transformers:
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            self.scalers['main'] = scaler
        else:
            if 'main' in self.scalers:
                X_scaled = self.scalers['main'].transform(X)
            else:
                X_scaled = X.values
        
        # Convert back to DataFrame
        X_scaled = pd.DataFrame(X_scaled, columns=X.columns, index=X.index)
        
        # Feature selection
        if fit_transformers and self.config.get('feature_selection', {}).get('enabled', False):
            k_best = self.config['feature_selection']['k_best']
            selector = SelectKBest(score_func=f_classif, k=k_best)
            X_selected = selector.fit_transform(X_scaled, y)
            self.feature_selectors['main'] = selector
            
            # Get selected feature names
            selected_features = X_scaled.columns[selector.get_support()].tolist()
            X_scaled = pd.DataFrame(X_selected, columns=selected_features, index=X_scaled.index)
        
        logger.info(f"Prepared {X_scaled.shape[1]} features for training")
        
        return X_scaled, y
    
    def transform_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transform new data using fitted transformers.
        
        Args:
            df: Input DataFrame
            
        Returns:
            Transformed DataFrame
        """
        logger.info("Transforming features for prediction")
        
        # Handle categorical variables
        categorical_columns = df.select_dtypes(include=['object', 'category']).columns
        
        for col in categorical_columns:
            if col in self.encoders:
                df[col] = self.encoders[col].transform(df[col].astype(str))
        
        # Handle missing values
        df = df.fillna(df.median())
        
        # Feature scaling
        if 'main' in self.scalers:
            df_scaled = self.scalers['main'].transform(df)
            df_scaled = pd.DataFrame(df_scaled, columns=df.columns, index=df.index)
        else:
            df_scaled = df
        
        # Feature selection
        if 'main' in self.feature_selectors:
            df_selected = self.feature_selectors['main'].transform(df_scaled)
            selected_features = df_scaled.columns[self.feature_selectors['main'].get_support()].tolist()
            df_scaled = pd.DataFrame(df_selected, columns=selected_features, index=df_scaled.index)
        
        return df_scaled
