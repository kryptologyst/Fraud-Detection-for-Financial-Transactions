"""Streamlit demo application for fraud detection.

This application provides an interactive interface for fraud detection
model evaluation and real-time scoring.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import joblib
from pathlib import Path
import sys
from typing import Dict, Any, Optional

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent / "src"))

from data.generator import TransactionDataGenerator, load_config
from features.engineering import FeatureEngineer
from risk.evaluation import FraudDetectionEvaluator

# Page configuration
st.set_page_config(
    page_title="Fraud Detection Research Demo",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .disclaimer-box {
        background-color: #ffebee;
        border: 2px solid #f44336;
        border-radius: 5px;
        padding: 1rem;
        margin: 1rem 0;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 5px;
        margin: 0.5rem 0;
    }
    .fraud-alert {
        background-color: #ffcdd2;
        border: 2px solid #d32f2f;
        border-radius: 5px;
        padding: 1rem;
        margin: 1rem 0;
    }
    .normal-transaction {
        background-color: #c8e6c9;
        border: 2px solid #388e3c;
        border-radius: 5px;
        padding: 1rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Disclaimer
st.markdown("""
<div class="disclaimer-box">
    <h3>⚠️ IMPORTANT DISCLAIMER</h3>
    <p><strong>This is a research and educational demonstration only.</strong></p>
    <ul>
        <li>NOT for investment advice or financial decision making</li>
        <li>NOT for production fraud detection systems</li>
        <li>Uses synthetic data that may not reflect real-world patterns</li>
        <li>Models may be inaccurate and should not be relied upon</li>
        <li>No guarantees about performance, accuracy, or reliability</li>
    </ul>
    <p><strong>Use at your own risk. See DISCLAIMER.md for full details.</strong></p>
</div>
""", unsafe_allow_html=True)

# Main header
st.markdown('<h1 class="main-header">🔍 Fraud Detection Research Demo</h1>', unsafe_allow_html=True)

# Sidebar
st.sidebar.title("Configuration")
st.sidebar.markdown("### Model Selection")

# Load models if they exist
models_dir = Path("assets")
models = {}
feature_engineer = None

if models_dir.exists():
    model_files = list(models_dir.glob("*_model.joblib"))
    if model_files:
        for model_file in model_files:
            model_name = model_file.stem.replace("_model", "")
            try:
                models[model_name] = joblib.load(model_file)
                st.sidebar.success(f"✅ Loaded {model_name} model")
            except Exception as e:
                st.sidebar.error(f"❌ Failed to load {model_name}: {str(e)}")
        
        # Load feature engineer
        fe_file = models_dir / "feature_engineer.joblib"
        if fe_file.exists():
            try:
                feature_engineer = joblib.load(fe_file)
                st.sidebar.success("✅ Loaded feature engineer")
            except Exception as e:
                st.sidebar.error(f"❌ Failed to load feature engineer: {str(e)}")
    else:
        st.sidebar.warning("⚠️ No trained models found. Please run training first.")
else:
    st.sidebar.warning("⚠️ Assets directory not found. Please run training first.")

# Model selection
if models:
    selected_model = st.sidebar.selectbox(
        "Select Model for Scoring",
        list(models.keys()),
        help="Choose which trained model to use for fraud detection"
    )
    
    # Model info
    model = models[selected_model]
    st.sidebar.markdown(f"**Selected Model:** {selected_model}")
    
    if hasattr(model, 'get_feature_importance'):
        feature_importance = model.get_feature_importance()
        if feature_importance is not None:
            st.sidebar.markdown("**Top 5 Features:**")
            for i, (feature, importance) in enumerate(feature_importance.head().items()):
                st.sidebar.text(f"{i+1}. {feature}: {importance:.4f}")

# Main content tabs
tab1, tab2, tab3, tab4 = st.tabs(["🎯 Real-time Scoring", "📊 Model Performance", "🔍 Feature Analysis", "📈 Data Insights"])

with tab1:
    st.header("Real-time Fraud Detection")
    
    if not models or not feature_engineer:
        st.error("Please train models first by running: `python scripts/train_models.py`")
        st.stop()
    
    st.markdown("Enter transaction details to get fraud probability score:")
    
    col1, col2 = st.columns(2)
    
    with col1:
        amount = st.number_input(
            "Transaction Amount ($)",
            min_value=0.01,
            max_value=10000.0,
            value=100.0,
            step=0.01,
            help="Amount of the transaction in dollars"
        )
        
        user_id = st.number_input(
            "User ID",
            min_value=1,
            max_value=1000,
            value=1,
            help="Unique identifier for the user"
        )
        
        hour = st.slider(
            "Transaction Hour",
            min_value=0,
            max_value=23,
            value=12,
            help="Hour of the day (0-23)"
        )
        
    with col2:
        merchant_category = st.selectbox(
            "Merchant Category",
            ["Grocery", "Gas Station", "Restaurant", "Online Shopping", "ATM", 
             "Pharmacy", "Entertainment", "Travel", "Healthcare", "Education"],
            help="Category of the merchant"
        )
        
        location = st.selectbox(
            "Location",
            ["New York", "Los Angeles", "Chicago", "Houston", "Phoenix", 
             "Philadelphia", "San Antonio", "San Diego", "Dallas", "San Jose"],
            help="Geographic location of the transaction"
        )
        
        device_type = st.selectbox(
            "Device Type",
            ["Mobile", "Desktop", "Tablet", "ATM"],
            help="Type of device used for the transaction"
        )
    
    # Create transaction data
    if st.button("🔍 Analyze Transaction", type="primary"):
        # Create transaction DataFrame
        transaction_data = {
            'transaction_id': [1],
            'amount': [amount],
            'user_id': [user_id],
            'timestamp': [pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')],
            'merchant_category': [merchant_category],
            'location': [location],
            'device_type': [device_type]
        }
        
        df_transaction = pd.DataFrame(transaction_data)
        
        # Engineer features
        df_features = feature_engineer.create_features(df_transaction)
        
        # Prepare features for prediction
        X_transaction = feature_engineer.transform_features(df_features)
        
        # Get prediction
        fraud_proba = model.predict_proba(X_transaction)
        if fraud_proba.shape[1] == 2:
            fraud_score = fraud_proba[0, 1]
        else:
            fraud_score = fraud_proba[0]
        
        # Display results
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(
                "Fraud Probability",
                f"{fraud_score:.1%}",
                delta=None
            )
        
        with col2:
            st.metric(
                "Risk Level",
                "HIGH" if fraud_score > 0.7 else "MEDIUM" if fraud_score > 0.3 else "LOW",
                delta=None
            )
        
        with col3:
            st.metric(
                "Recommendation",
                "INVESTIGATE" if fraud_score > 0.5 else "MONITOR" if fraud_score > 0.2 else "APPROVE",
                delta=None
            )
        
        # Visual representation
        fig = go.Figure(go.Indicator(
            mode = "gauge+number+delta",
            value = fraud_score * 100,
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': "Fraud Risk Score"},
            delta = {'reference': 50},
            gauge = {
                'axis': {'range': [None, 100]},
                'bar': {'color': "darkblue"},
                'steps': [
                    {'range': [0, 30], 'color': "lightgray"},
                    {'range': [30, 70], 'color': "yellow"},
                    {'range': [70, 100], 'color': "red"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 70
                }
            }
        ))
        
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
        
        # Alert box
        if fraud_score > 0.7:
            st.markdown("""
            <div class="fraud-alert">
                <h3>🚨 HIGH FRAUD RISK DETECTED</h3>
                <p>This transaction shows characteristics consistent with fraudulent activity. 
                Recommend immediate investigation and potential transaction blocking.</p>
            </div>
            """, unsafe_allow_html=True)
        elif fraud_score > 0.3:
            st.markdown("""
            <div class="metric-card">
                <h3>⚠️ MEDIUM FRAUD RISK</h3>
                <p>This transaction shows some suspicious characteristics. 
                Recommend additional verification or monitoring.</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="normal-transaction">
                <h3>✅ LOW FRAUD RISK</h3>
                <p>This transaction appears to be legitimate based on current patterns. 
                Normal processing recommended.</p>
            </div>
            """, unsafe_allow_html=True)

with tab2:
    st.header("Model Performance Analysis")
    
    # Load evaluation results if available
    results_file = Path("assets/model_comparison.csv")
    if results_file.exists():
        comparison_df = pd.read_csv(results_file)
        
        st.subheader("Model Comparison")
        st.dataframe(comparison_df, use_container_width=True)
        
        # Performance metrics visualization
        metrics = ['ROC AUC', 'PR AUC', 'Precision', 'Recall', 'F1 Score']
        
        fig = make_subplots(
            rows=2, cols=3,
            subplot_titles=metrics,
            specs=[[{'type': 'bar'}, {'type': 'bar'}, {'type': 'bar'}],
                   [{'type': 'bar'}, {'type': 'bar'}, {'type': 'scatter'}]]
        )
        
        for i, metric in enumerate(metrics):
            row = i // 3 + 1
            col = i % 3 + 1
            
            fig.add_trace(
                go.Bar(
                    x=comparison_df['Model'],
                    y=comparison_df[metric],
                    name=metric,
                    showlegend=False
                ),
                row=row, col=col
            )
        
        fig.update_layout(height=600, title_text="Model Performance Comparison")
        st.plotly_chart(fig, use_container_width=True)
        
        # Best model
        best_model = comparison_df.loc[comparison_df['ROC AUC'].idxmax()]
        st.success(f"🏆 Best performing model: **{best_model['Model']}** (ROC AUC: {best_model['ROC AUC']:.4f})")
        
    else:
        st.warning("No evaluation results found. Please run training first.")

with tab3:
    st.header("Feature Analysis")
    
    if models and feature_engineer:
        # Feature importance
        if hasattr(model, 'get_feature_importance'):
            feature_importance = model.get_feature_importance()
            if feature_importance is not None:
                st.subheader("Feature Importance")
                
                # Top 20 features
                top_features = feature_importance.head(20)
                
                fig = go.Figure(go.Bar(
                    x=top_features.values,
                    y=top_features.index,
                    orientation='h',
                    marker_color='lightblue'
                ))
                
                fig.update_layout(
                    title="Top 20 Most Important Features",
                    xaxis_title="Importance Score",
                    yaxis_title="Feature",
                    height=600
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Feature importance table
                st.subheader("Feature Importance Details")
                st.dataframe(top_features.to_frame('Importance'), use_container_width=True)
            else:
                st.info("Feature importance not available for the selected model.")
        else:
            st.info("Feature importance not available for the selected model.")
    else:
        st.warning("Please train models first.")

with tab4:
    st.header("Data Insights")
    
    # Load data if available
    data_file = Path("assets/raw_transactions.csv")
    if data_file.exists():
        df = pd.read_csv(data_file)
        
        st.subheader("Dataset Overview")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Transactions", f"{len(df):,}")
        with col2:
            st.metric("Fraud Cases", f"{df['is_fraud'].sum():,}")
        with col3:
            st.metric("Fraud Rate", f"{df['is_fraud'].mean():.1%}")
        with col4:
            st.metric("Unique Users", f"{df['user_id'].nunique():,}")
        
        # Transaction amount distribution
        st.subheader("Transaction Amount Distribution")
        fig = go.Figure()
        fig.add_trace(go.Histogram(
            x=df['amount'],
            nbinsx=50,
            name='All Transactions'
        ))
        fig.add_trace(go.Histogram(
            x=df[df['is_fraud']==1]['amount'],
            nbinsx=50,
            name='Fraudulent Transactions'
        ))
        
        fig.update_layout(
            title="Transaction Amount Distribution",
            xaxis_title="Amount ($)",
            yaxis_title="Frequency",
            barmode='overlay'
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Fraud by merchant category
        st.subheader("Fraud Rate by Merchant Category")
        fraud_by_merchant = df.groupby('merchant_category')['is_fraud'].agg(['count', 'sum', 'mean']).reset_index()
        fraud_by_merchant.columns = ['Merchant Category', 'Total Transactions', 'Fraud Cases', 'Fraud Rate']
        fraud_by_merchant = fraud_by_merchant.sort_values('Fraud Rate', ascending=False)
        
        fig = go.Figure(go.Bar(
            x=fraud_by_merchant['Merchant Category'],
            y=fraud_by_merchant['Fraud Rate'],
            marker_color='red'
        ))
        
        fig.update_layout(
            title="Fraud Rate by Merchant Category",
            xaxis_title="Merchant Category",
            yaxis_title="Fraud Rate",
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Fraud by hour
        st.subheader("Fraud Rate by Hour of Day")
        df['hour'] = pd.to_datetime(df['timestamp']).dt.hour
        fraud_by_hour = df.groupby('hour')['is_fraud'].agg(['count', 'sum', 'mean']).reset_index()
        fraud_by_hour.columns = ['Hour', 'Total Transactions', 'Fraud Cases', 'Fraud Rate']
        
        fig = go.Figure(go.Scatter(
            x=fraud_by_hour['Hour'],
            y=fraud_by_hour['Fraud Rate'],
            mode='lines+markers',
            line=dict(color='red', width=3),
            marker=dict(size=8)
        ))
        
        fig.update_layout(
            title="Fraud Rate by Hour of Day",
            xaxis_title="Hour of Day",
            yaxis_title="Fraud Rate",
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
    else:
        st.warning("No data found. Please run training first to generate sample data.")

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; font-size: 0.8rem;">
    <p>🔍 Fraud Detection Research Demo | For Educational Purposes Only | 
    <a href="DISCLAIMER.md" target="_blank">Full Disclaimer</a></p>
</div>
""", unsafe_allow_html=True)
