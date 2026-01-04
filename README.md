# Fraud Detection for Financial Transactions

## IMPORTANT DISCLAIMER

**RESEARCH AND EDUCATIONAL PURPOSES ONLY**

This project is designed exclusively for research and educational purposes. It is NOT intended for investment advice, financial decision making, or production fraud detection systems.

**LIMITATIONS AND RISKS:**
- Models may be inaccurate and should not be relied upon for actual fraud detection
- Uses simulated data that may not reflect real-world transaction patterns
- No guarantees about performance, accuracy, or reliability
- Does not meet regulatory requirements for financial institutions

**USE AT YOUR OWN RISK** - See [DISCLAIMER.md](DISCLAIMER.md) for full details.

---

## Overview

This project implements a comprehensive fraud detection system for financial transactions using advanced machine learning techniques. The system includes multiple models, proper evaluation metrics, and an interactive demo for research and educational purposes.

## Features

- **Multiple ML Models**: Random Forest, XGBoost, LightGBM, Isolation Forest, Autoencoders
- **Advanced Evaluation**: Fraud-specific metrics (Precision@K, AUCPR, KS statistic)
- **Explainability**: SHAP explanations and feature importance analysis
- **Risk Management**: Uncertainty quantification and model governance
- **Interactive Demo**: Streamlit-based web interface for real-time scoring
- **Production-Ready**: Proper structure, configs, tests, and documentation

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/kryptologyst/Fraud-Detection-for-Financial-Transactions.git
cd Fraud-Detection-for-Financial-Transactions

# Install dependencies
pip install -e .

# Install development dependencies (optional)
pip install -e ".[dev]"
```

### Basic Usage

```bash
# Generate synthetic data and train models
python scripts/train_models.py

# Run evaluation
python scripts/evaluate_models.py

# Launch interactive demo
streamlit run demo/app.py
```

## Project Structure

```
├── src/                    # Source code
│   ├── data/              # Data processing and generation
│   ├── features/          # Feature engineering
│   ├── models/            # ML models and training
│   ├── backtest/          # Backtesting framework
│   ├── risk/              # Risk management and evaluation
│   └── utils/             # Utility functions
├── configs/               # Configuration files
├── scripts/               # Training and evaluation scripts
├── notebooks/             # Jupyter notebooks for analysis
├── tests/                 # Unit tests
├── assets/                # Generated plots and results
├── demo/                  # Streamlit demo application
└── data/                  # Data storage (if using real data)
```

## Models and Techniques

### Supervised Learning
- **Random Forest**: Baseline ensemble method
- **XGBoost**: Gradient boosting with fraud-specific optimizations
- **LightGBM**: Fast gradient boosting with categorical features
- **Neural Networks**: Deep learning for complex patterns

### Unsupervised Learning
- **Isolation Forest**: Anomaly detection for unknown fraud types
- **Autoencoders**: Deep learning anomaly detection
- **One-Class SVM**: Support vector machines for outlier detection

### Feature Engineering
- Transaction amount statistics and distributions
- Temporal features (hour, day, month patterns)
- User behavior patterns and deviations
- Cross-feature interactions and ratios

## Evaluation Metrics

### Classification Metrics
- **AUROC**: Area under ROC curve
- **AUCPR**: Area under Precision-Recall curve
- **Precision@K**: Precision at top K predictions
- **KS Statistic**: Kolmogorov-Smirnov test for score distributions

### Business Metrics
- **False Positive Rate**: Legitimate transactions flagged as fraud
- **Detection Rate**: Percentage of fraud cases caught
- **Investigation Efficiency**: Cases requiring manual review

## Configuration

The system uses OmegaConf for configuration management. Key configs:

- `configs/data.yaml`: Data generation and processing parameters
- `configs/models.yaml`: Model hyperparameters and training settings
- `configs/evaluation.yaml`: Evaluation metrics and thresholds

## Demo Application

The Streamlit demo provides:

1. **Real-time Scoring**: Input transaction details and get fraud probability
2. **Model Comparison**: Compare different models side-by-side
3. **Feature Analysis**: SHAP explanations and feature importance
4. **Performance Metrics**: Live evaluation of model performance
5. **Risk Assessment**: Uncertainty quantification and confidence intervals

## Development

### Code Quality
- **Type Hints**: Full type annotations for better code clarity
- **Documentation**: Google-style docstrings for all functions
- **Testing**: Comprehensive unit tests with pytest
- **Linting**: Black formatting and Ruff linting
- **Pre-commit**: Automated code quality checks

### Running Tests
```bash
pytest tests/
```

### Code Formatting
```bash
black src/ tests/
ruff check src/ tests/
```

## Data Schema

### Transaction Data
- `transaction_id`: Unique identifier
- `amount`: Transaction amount
- `timestamp`: Transaction timestamp
- `user_id`: User identifier
- `merchant_category`: Merchant category code
- `location`: Geographic location
- `device_info`: Device fingerprint
- `is_fraud`: Fraud label (0/1)

### Generated Features
- Statistical features (mean, std, percentiles)
- Temporal features (hour, day patterns)
- Behavioral features (frequency, amounts)
- Cross-feature interactions

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes with proper tests
4. Ensure code quality standards
5. Submit a pull request

## License

MIT License - see LICENSE file for details.

## Citation

If you use this code in your research, please cite:

```bibtex
@software{fraud_detection_research,
  title={Fraud Detection for Financial Transactions},
  author={Kryptologyst},
  year={2026},
  url={https://github.com/kryptologyst/Fraud-Detection-for-Financial-Transactions}
}
```

## Support

For questions and support:
- Create an issue on GitHub
- Check the documentation in `notebooks/`
- Review the demo application for usage examples

---

**Remember: This is for research and educational purposes only. Do not use for actual financial decisions or production fraud detection systems.**
# Fraud-Detection-for-Financial-Transactions
