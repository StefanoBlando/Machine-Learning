# 🤖 Advanced Income Classification

A comprehensive machine learning pipeline for binary income classification utilizing advanced feature engineering, hyperparameter optimization, and ensemble methods.

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.3+-orange.svg)](https://scikit-learn.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-Latest-brightgreen.svg)](https://xgboost.readthedocs.io/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Code Style](https://img.shields.io/badge/Code%20Style-Black-black.svg)](https://black.readthedocs.io/)

## 🎯 Project Objective

Develop a robust binary classification system to predict whether an individual's income exceeds $50K, implementing state-of-the-art machine learning techniques including:

- **Advanced Feature Engineering** with domain expertise
- **Multiple Class Imbalance Strategies** (SMOTE, ADASYN, SMOTEENN)
- **Bayesian Hyperparameter Optimization** with Optuna integration
- **Ensemble Methods** (Voting, Stacking, Weighted)
- **Model Interpretability** using SHAP and Permutation Importance
- **Business-Oriented Threshold Optimization** for profit maximization

## 🏆 Key Results

| Model Configuration | F1-Score | ROC-AUC | Business Profit | Training Strategy |
|-------------------|----------|---------|----------------|-------------------|
| **Best Individual** | **0.xxxx** | **0.xxxx** | **$xxx,xxx** | XGBoost + SMOTE |
| **Best Ensemble** | **0.xxxx** | **0.xxxx** | **$xxx,xxx** | Stacking Classifier |
| Traditional Baseline | 0.xxxx | 0.xxxx | $xxx,xxx | Random Forest |

*Results based on 70/30 stratified split with optimal threshold optimization*

## 🔬 Model Architecture

### Traditional Models (Core Requirements)
- **Logistic Regression** - L1/L2 regularization with balanced class weights
- **Random Forest** - 100-500 estimators with feature importance analysis
- **Gradient Boosting** - Sequential learning with subsample optimization
- **Support Vector Machine** - RBF/Linear kernels with probability calibration

### Advanced Models (Performance Enhancement)
- **XGBoost** - Extreme gradient boosting with scale_pos_weight balancing
- **LightGBM** - Microsoft's efficient gradient boosting framework

### Ensemble Methods (Performance Maximization)
- **Voting Classifier** - Soft voting across top-performing models
- **Stacking Classifier** - Meta-learner with cross-validation
- **Weighted Average** - Performance-weighted probability aggregation

## 🚀 Quick Start

### Installation

```bash
# Clone repository
git clone https://github.com/yourusername/advanced-income-classification.git
cd advanced-income-classification

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install package in development mode
pip install -e .
```

### Basic Usage

```python
from src.data.loader import load_and_explore_data
from src.data.feature_engineering import advanced_feature_engineering
from src.preprocessing.pipeline import advanced_preprocessing_and_split
from src.models.traditional import setup_traditional_models
from src.optimization.hyperparameter_tuning import perform_hyperparameter_optimization

# Load and engineer features
df = load_and_explore_data('data/raw/data.csv')
df_engineered = advanced_feature_engineering(df, target_col='target')

# Preprocess and split
preprocessing_results = advanced_preprocessing_and_split(
    df_engineered, target_col='target', test_size=0.3, random_state=123
)

# Train models
models_config = setup_traditional_models()
results = perform_hyperparameter_optimization(
    models_config, 
    preprocessing_results['balanced_datasets'],
    preprocessing_results['cv_strategies']
)
```

### Command Line Interface

```bash
# Train all models with default configuration
python scripts/run_complete_pipeline.py --data-path data/raw/data.csv

# Train specific model type
python scripts/train_traditional_models.py --data-path data/raw/data.csv --output-dir results/

# Evaluate saved models
python scripts/evaluate_models.py --models-dir models/saved_models/ --test-data data/raw/data.csv

# Generate model card
python scripts/generate_model_card.py --best-model models/saved_models/best_model.pkl
```

## 📊 Data Pipeline

### Feature Engineering (20+ New Features)
- **Occupation Categorization** - Grouped into 5 business-relevant categories
- **Age-based Features** - Life stage groups and polynomial transformations
- **Work Pattern Analysis** - Intensity scoring and overtime indicators
- **Financial Behavior** - Capital gain/loss ratios and log transformations
- **Interaction Features** - Age×Education, work efficiency metrics
- **Target Encoding** - Group-based income rate features

### Preprocessing Pipeline
- **Missing Value Imputation** - Mode for categorical, median for numerical
- **Feature Selection** - SelectKBest with f_classif scoring
- **Scaling** - StandardScaler for numerical features
- **Encoding** - OneHotEncoder for categorical features with unknown handling
- **Class Balancing** - Multiple strategies (SMOTE, ADASYN, SMOTEENN)

## 🔧 Advanced Techniques

### Hyperparameter Optimization
- **BayesSearchCV** - Efficient Bayesian optimization
- **Optuna Framework** - TPE, CMA-ES, Random samplers with pruning
- **Nested Cross-Validation** - Unbiased performance estimation
- **Smart Iteration Allocation** - Model-specific optimization budgets

### Model Interpretability
- **SHAP Analysis** - TreeExplainer and KernelExplainer
- **Permutation Importance** - Model-agnostic feature ranking
- **Feature Importance** - Native model importance scores
- **Stability Analysis** - Bootstrap performance validation

### Business Optimization
- **Cost-Sensitive Metrics** - Custom profit maximization
- **Threshold Optimization** - ROC curve analysis for business impact
- **Performance Tracking** - F1, ROC-AUC, MCC, Balanced Accuracy

## 📈 Evaluation Framework

### Cross-Validation Strategy
- **Stratified K-Fold** - 5-fold with 3 repeats
- **Train/Validation Split** - 70/30 stratified (seed=123)
- **Bootstrap Validation** - Model stability assessment
- **Nested CV** - Robust generalization estimation

### Comprehensive Metrics
```python
{
    'f1_score': 0.xxxx,           # Primary optimization metric
    'roc_auc': 0.xxxx,            # Discrimination capability
    'accuracy': 0.xxxx,           # Overall correctness
    'precision': 0.xxxx,          # Positive predictive value
    'recall': 0.xxxx,             # Sensitivity
    'balanced_accuracy': 0.xxxx,  # Class-balanced performance
    'matthews_corrcoef': 0.xxxx,  # Correlation coefficient
    'business_profit': $xxx,xxx   # Economic impact
}
```

## 📁 Repository Structure

```
advanced-income-classification/
├── src/                          # Source code modules
│   ├── data/                    # Data loading and feature engineering
│   ├── models/                  # Model implementations
│   ├── optimization/            # Hyperparameter tuning
│   ├── evaluation/             # Metrics and interpretability
│   └── visualization/          # Plotting and analysis
├── notebooks/                   # Jupyter analysis notebooks
├── scripts/                     # Command-line tools
├── tests/                       # Unit and integration tests
├── data/                        # Dataset storage
├── models/                      # Saved models and artifacts
└── results/                     # Experiment outputs
```

## 🐳 Docker Support

```bash
# Build image
docker build -t income-classification .

# Run training pipeline
docker run -v $(pwd)/data:/app/data income-classification python scripts/run_complete_pipeline.py

# Run with GPU support (if available)
docker run --gpus all -v $(pwd)/data:/app/data income-classification
```

## 🧪 Testing

```bash
# Run all tests
pytest tests/

# Run specific test module
pytest tests/test_feature_engineering.py -v

# Run with coverage
pytest tests/ --cov=src --cov-report=html
```

## 📚 Documentation

- **[Installation Guide](docs/installation.md)** - Detailed setup instructions
- **[Methodology](docs/methodology.md)** - Technical approach and decisions
- **[API Reference](docs/api_reference.md)** - Function and class documentation
- **[Performance Analysis](docs/performance_analysis.md)** - Model comparison and insights

## 🛠️ Development

### Setting up Development Environment

```bash
# Install development dependencies
pip install -r requirements-dev.txt

# Install pre-commit hooks
pre-commit install

# Run code formatting
black src/ tests/
flake8 src/ tests/

# Type checking
mypy src/
```

### Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the AGPL License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **UCI Machine Learning Repository** for the Adult Census Income dataset
- **Scikit-learn** community for the comprehensive ML framework
- **Optuna** team for advanced hyperparameter optimization
- **SHAP** developers for model interpretability tools

]
- **Project Link**: [https://github.com/yourusername/advanced-income-classification](https://github.com/yourusername/advanced-income-classification)

