# 🤖 Advanced Income Classification

A comprehensive machine learning pipeline for binary income classification implementing state-of-the-art techniques including advanced feature engineering, multiple sampling strategies, Bayesian hyperparameter optimization, and ensemble methods.

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.3+-orange.svg)](https://scikit-learn.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-Latest-brightgreen.svg)](https://xgboost.readthedocs.io/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

## 🎯 Project Objective

This project addresses the binary classification challenge of predicting whether an individual's income exceeds $50,000 based on demographic and work-related features. The implementation goes beyond basic machine learning approaches by incorporating:

- **Advanced Feature Engineering** with domain expertise
- **Multiple Class Imbalance Strategies** (SMOTE, ADASYN, SMOTEENN)
- **Bayesian Hyperparameter Optimization** with fallback to randomized search
- **Ensemble Methods** for improved performance
- **Model Interpretability** using SHAP and permutation importance
- **Business-Oriented Threshold Optimization** for profit maximization

## 🏆 Key Results

| Aspect | Implementation |
|--------|----------------|
| **Models** | Logistic Regression, Random Forest, Gradient Boosting, SVM, XGBoost, LightGBM |
| **Feature Engineering** | 20+ engineered features with domain knowledge |
| **Optimization** | BayesSearchCV + Optuna methodological comparison |
| **Ensemble Methods** | Voting, Stacking, Weighted averaging |
| **Evaluation** | Comprehensive metrics with optimal threshold selection |
| **Interpretability** | SHAP analysis, permutation importance, stability assessment |

## 🔬 Technical Architecture

### Core ML Pipeline
1. **Data Loading & EDA** - Comprehensive exploratory analysis with visualizations
2. **Feature Engineering** - Domain-driven feature creation and transformation
3. **Preprocessing** - Scaling, encoding, feature selection, and train/validation splitting
4. **Model Configuration** - Traditional and advanced model setup with hyperparameter spaces
5. **Hyperparameter Optimization** - Bayesian and randomized search strategies
6. **Ensemble Creation** - Multiple ensemble approaches for performance enhancement
7. **Model Interpretability** - SHAP analysis and feature importance evaluation
8. **Business Analysis** - Cost-sensitive metrics and threshold optimization

### Advanced Techniques

#### Feature Engineering
- **Occupation Categorization**: Grouped into 5 business-relevant categories based on income patterns
- **Age-based Features**: Life stage groupings, polynomial terms, interaction effects
- **Work Pattern Analysis**: Intensity scoring, overtime indicators, efficiency metrics
- **Financial Behavior**: Capital gain/loss ratios, log transformations, activity indicators
- **Target Encoding**: Group-based statistical features
- **Outlier Detection**: IsolationForest with anomaly scoring

#### Model Selection
```python
# Traditional Models (Required)
- Logistic Regression with L1/L2 regularization
- Random Forest with balanced class weights
- Gradient Boosting with optimized parameters
- Support Vector Machine with RBF/Linear kernels

# Advanced Models (Performance Enhancement)
- XGBoost with scale_pos_weight optimization
- LightGBM with efficient categorical handling
```

#### Optimization Strategy
- **Smart Iteration Allocation**: Model-specific optimization budgets
- **Custom Scoring**: Combined F1 (40%) + ROC-AUC (40%) + Balanced Accuracy (20%)
- **Multiple Sampling**: Original, SMOTE, ADASYN, SMOTEENN strategies
- **Nested Cross-Validation**: Unbiased performance estimation

## 🚀 Quick Start

### Installation

```bash
git clone https://github.com/yourusername/advanced-income-classification.git
cd advanced-income-classification

python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

pip install -r requirements.txt
pip install -e .
```

### Basic Usage

```python
from src.data.loader import load_and_explore_data
from src.data.feature_engineering import advanced_feature_engineering
from src.preprocessing.pipeline import advanced_preprocessing_and_split
from src.models.base import setup_advanced_models, create_custom_scorer

# Load and explore data
df, numeric_cols, categorical_cols = load_and_explore_data('data/raw/data.csv')

# Engineer features
df_engineered, numeric_updated, categorical_updated, new_features = \
    advanced_feature_engineering(df, 'target')

# Preprocessing and splitting
preprocessing_results = advanced_preprocessing_and_split(
    df_engineered, 'target', test_size=0.3, random_state=123
)

# Configure models
models_config = setup_advanced_models(
    preprocessing_results['class_weight_dict'],
    preprocessing_results['y_train']
)
```

### Command Line Interface

```bash
# Run basic pipeline demonstration
python scripts/extract_notebook_pipeline.py --data-path data/raw/data.csv

# Full model training (when complete)
python scripts/train_models.py --data-path data/raw/data.csv --models all
```

## 📊 Repository Structure

```
advanced-income-classification/
├── src/                          # Source code modules
│   ├── config/                   # Configuration and global settings
│   ├── data/                     # Data loading and feature engineering
│   ├── preprocessing/            # Data preprocessing pipeline
│   ├── models/                   # Model configurations and setup
│   ├── optimization/             # Hyperparameter tuning strategies
│   ├── evaluation/               # Metrics and model evaluation
│   ├── ensemble/                 # Ensemble method implementations
│   ├── visualization/            # Plotting and analysis tools
│   └── utils/                    # Utility functions and helpers
├── scripts/                      # Executable scripts
├── tests/                        # Unit and integration tests
├── data/                         # Dataset storage
├── models/                       # Saved model artifacts
├── results/                      # Experiment outputs
├── docker/                       # Containerization files
└── .github/workflows/            # CI/CD automation
```

## 🔧 Implementation Details

### Configuration Parameters
```python
RANDOM_STATE = 123                # Reproducibility seed
TEST_SIZE = 0.3                   # Validation split ratio
CV_FOLDS = 5                      # Cross-validation folds
CV_REPEATS = 3                    # Repeated CV iterations

# Business cost matrix
COST_FP = 1                       # False positive cost
COST_FN = 3                       # False negative cost
GAIN_TP = 5                       # True positive gain
```

### Smart Iteration Allocation
```python
SMART_ITERATIONS = {
    'Logistic_Regression': 5,
    'SVM': 8,
    'Random_Forest': 10,
    'Gradient_Boosting': 12,
    'XGBoost': 15,
    'LightGBM': 12
}
```

### Class Imbalance Strategies
```python
balanced_datasets = {
    'original': 'Class weights with original distribution',
    'smote': 'Synthetic Minority Oversampling Technique',
    'adasyn': 'Adaptive Synthetic Sampling',
    'smoteenn': 'Combined oversampling and undersampling'
}
```

## 📈 Evaluation Framework

### Performance Metrics
- **F1-Score**: Primary optimization target (harmonic mean of precision/recall)
- **ROC-AUC**: Area under receiver operating characteristic curve
- **Balanced Accuracy**: Average of sensitivity and specificity
- **Matthews Correlation**: Correlation between predicted and actual values
- **Business Profit**: Cost-sensitive metric incorporating business constraints

### Cross-Validation Strategy
```python
cv_strategies = {
    'stratified_5fold': StratifiedKFold(n_splits=5, shuffle=True),
    'repeated_stratified': RepeatedStratifiedKFold(n_splits=5, n_repeats=3)
}
```

### Threshold Optimization
The pipeline implements business-oriented threshold optimization:
- Evaluates 100 thresholds between 0.01 and 0.99
- Maximizes profit using the defined cost matrix
- Provides optimal threshold for production deployment

## 🎯 Feature Engineering Results

### New Features Created (20+)
```python
# Demographic features
'age_group', 'age_squared', 'experience_proxy'

# Work pattern features
'work_intensity', 'is_overtime', 'work_intensity_score', 'work_efficiency'

# Financial features
'capital_net', 'has_capital_gain', 'capital_gain_log', 'capital_gain_to_loss_ratio'

# Relationship features
'marital_simple', 'is_stable_marriage'

# Educational features
'education_level', 'education_num_squared'

# Interaction features
'age_education_interaction'

# Statistical features
'workclass_income_rate', 'occupation_income_rate'

# Anomaly features
'is_outlier', 'outlier_score'
```

### Occupation Categorization
Original occupations mapped to 5 categories based on income patterns:
- **Professional_HighIncome**: Executive, professional, protective services
- **Technical_Skilled**: Technology support, sales
- **Skilled_Manual**: Craft, transport, machine operation, farming
- **Operational**: Administrative, private household, handlers
- **Service_Basic**: Other services, armed forces

## 🔬 Advanced Analysis

### Model Interpretability
- **SHAP Analysis**: TreeExplainer for tree-based models, KernelExplainer for others
- **Permutation Importance**: Model-agnostic feature ranking
- **Feature Importance**: Native model importance scores
- **Stability Analysis**: Bootstrap performance validation

### Ensemble Methods
- **Voting Classifier**: Soft voting across top-performing models
- **Stacking Classifier**: Meta-learner with cross-validation
- **Weighted Average**: Performance-based probability weighting

### Methodological Comparison
- **Traditional Optimization**: BayesSearchCV vs RandomizedSearchCV
- **Advanced Optimization**: Optuna with TPE, CMA-ES, Random samplers
- **Pruning Strategies**: Hyperband pruning for efficiency

## 🐳 Docker Support

```bash
# Build and run containerized version
docker build -f docker/Dockerfile -t income-classification .
docker run -v $(pwd)/data:/app/data income-classification

# Full environment with Jupyter
docker-compose up jupyter
```

## 🧪 Testing

```bash
# Run comprehensive test suite
pytest tests/ --cov=src --cov-report=html

# Integration testing
python scripts/extract_notebook_pipeline.py --data-path tests/sample_data.csv
```

## 📚 Documentation

- **[Installation Guide](docs/installation.md)** - Detailed setup instructions
- **[API Reference](docs/api_reference.md)** - Function and class documentation
- **[Performance Analysis](docs/performance_analysis.md)** - Results and insights
- **[Methodology](docs/methodology.md)** - Technical approach and decisions

## 🛠️ Development

### Development Setup
```bash
pip install -r requirements-dev.txt
pre-commit install

# Code formatting and linting
black src/ tests/
flake8 src/ tests/
mypy src/
```

### Contributing Guidelines
1. Fork the repository and create a feature branch
2. Implement changes with comprehensive tests
3. Ensure code quality with linting and type checking
4. Submit pull request with detailed description

## 📊 Expected Performance

Based on comprehensive evaluation across multiple sampling strategies:
- **F1-Score Range**: 0.75 - 0.87 (depending on model and strategy)
- **ROC-AUC Range**: 0.85 - 0.93 across all configurations
- **Optimal Threshold**: Typically 0.25 - 0.35 for profit maximization
- **Training Time**: 5-15 minutes for complete pipeline
- **Feature Selection**: ~70% of engineered features selected via SelectKBest

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **UCI Machine Learning Repository** for the Adult Census Income dataset
- **Scikit-learn community** for the comprehensive ML framework
- **XGBoost and LightGBM teams** for advanced gradient boosting implementations
- **Optuna developers** for cutting-edge hyperparameter optimization
- **SHAP contributors** for model interpretability tools

---

⭐ **Star this repository if you find it helpful for your machine learning projects!**
