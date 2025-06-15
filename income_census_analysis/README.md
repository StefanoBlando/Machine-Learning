# Income Census Analysis

A comprehensive machine learning project for binary classification to predict whether an individual's income exceeds $50K based on census data.

## 🎯 Project Overview

This project implements and compares four different machine learning algorithms to classify income levels using demographic and employment data from the Adult Census dataset. The goal is to build robust predictive models and identify the most effective approach for income classification.

## 🔬 Machine Learning Models

- **Random Forest Classifier**
- **Gradient Boosting Classifier** 
- **Support Vector Machine (SVM)**
- **Logistic Regression**

Each model undergoes hyperparameter optimization to achieve optimal performance.

## 📊 Dataset Information

- **Target Variable**: Binary classification (>50K, <=50K)
- **Features**: Age, work class, education level, marital status, occupation, relationship, race, gender, capital gain/loss, hours per week
- **Data Split**: 70% training, 30% validation (random seed: 123)
- **Preprocessing**: Feature engineering with occupation categories reduced to 5 main groups

## 🚀 Key Features

- **Comprehensive EDA**: Detailed exploratory data analysis with visualizations
- **Feature Engineering**: Strategic preprocessing and categorical variable handling
- **Model Comparison**: Side-by-side performance evaluation of all algorithms
- **Hyperparameter Tuning**: Systematic optimization using grid search
- **Performance Metrics**: Accuracy, Precision, Recall, F1-Score, ROC-AUC
- **Reproducible Results**: Fixed random seeds and version-controlled experiments

## 📁 Project Structure

```
income_census_analysis/
├── data/                    # Dataset storage
├── notebooks/              # Jupyter notebooks for analysis
├── src/                    # Source code modules
├── models/                 # Trained model artifacts
├── results/               # Outputs and visualizations
├── scripts/               # Automation scripts
├── config/                # Configuration files
└── docs/                  # Documentation
```

## 🛠️ Installation & Setup

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/income_census_analysis.git
cd income_census_analysis
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Run the analysis**
```bash
python scripts/train_models.py
```

## 📈 Usage

### Quick Start
```python
from src.models.model_trainer import ModelTrainer
from src.data.preprocess import DataPreprocessor

# Load and preprocess data
preprocessor = DataPreprocessor()
X_train, X_val, y_train, y_val = preprocessor.prepare_data()

# Train models
trainer = ModelTrainer()
results = trainer.train_all_models(X_train, y_train, X_val, y_val)
```

### Jupyter Notebooks
Explore the step-by-step analysis:
1. `01_exploratory_data_analysis.ipynb` - Data exploration and visualization
2. `02_data_preprocessing.ipynb` - Data cleaning and feature engineering
3. `03_feature_engineering.ipynb` - Advanced feature creation
4. `04_model_training.ipynb` - Model development and training
5. `05_model_evaluation.ipynb` - Performance comparison and analysis
6. `06_final_analysis.ipynb` - Results summary and insights

## 📊 Results

The project generates comprehensive performance comparisons including:
- Cross-validation scores for all models
- Feature importance analysis
- ROC curves and confusion matrices
- Hyperparameter optimization results
- Final model recommendations

Results are automatically saved to `results/reports/model_comparison.html`

## 🔧 Configuration

Model parameters and data processing options can be customized in `config/config.yaml`:

```yaml
data:
  train_size: 0.7
  random_state: 123
  
models:
  random_forest:
    n_estimators: [100, 200, 300]
    max_depth: [10, 20, None]
```

## 🧪 Testing

Run the test suite:
```bash
pytest tests/
```

## 📚 Dependencies

- Python 3.8+
- scikit-learn
- pandas
- numpy
- matplotlib
- seaborn
- jupyter

See `requirements.txt` for complete dependency list.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/improvement`)
3. Commit changes (`git commit -am 'Add new feature'`)
4. Push to branch (`git push origin feature/improvement`)
5. Create Pull Request

## 📄 License

This project is licensed under the GNU Affero General Public License v3.0 - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Adult Census dataset from UCI Machine Learning Repository
- Scikit-learn community for excellent ML tools
- Open source contributors who made this project possible
