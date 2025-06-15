# Income Census Analysis

A binary classification project to predict whether an individual's income exceeds $50K based on census data using advanced machine learning techniques.

## 🎯 Project Objective

Develop and compare multiple machine learning models to classify income levels using:
- **Random Forest**
- **Gradient Boosting** 
- **Support Vector Machine (SVM)**
- **Logistic Regression**

## 📊 Dataset Overview

- **Target Variable**: >50K, <=50K (income level)
- **Features**: age, workclass, education, marital status, occupation, relationship, race, sex, capital gains/losses, hours per week, native country
- **Total Records**: 32,561 observations
- **Features**: 15 columns

### Key Preprocessing Steps:
- Remove `education` and `native-country` columns
- Combine occupation categories into 5 groups
- Train/Validation split: 70%/30% (random_state=123)
- Hyperparameter optimization for all models

## 🚀 Quick Start

### Prerequisites
```bash
Python 3.8+
pip or conda package manager
```

### Installation
```bash
# Clone the repository
git clone https://github.com/username/income_census_analysis.git
cd income_census_analysis

# Install dependencies
pip install -r requirements.txt

# Or using conda
conda env create -f environment.yml
conda activate income-census-env
```

### Usage
```bash
# Run complete analysis pipeline
python scripts/train_models.py

# Run hyperparameter tuning
python scripts/hyperparameter_tuning.py

# Generate evaluation report
python scripts/evaluate_models.py

# Create visualizations
python scripts/generate_report.py
```

## 📁 Project Structure

```
income_census_analysis/
├── data/                    # Data files
│   ├── raw/                # Original dataset
│   ├── processed/          # Cleaned and processed data
│   └── external/           # External data sources
├── notebooks/              # Jupyter notebooks for analysis
├── src/                    # Source code modules
│   ├── data/              # Data processing modules
│   ├── models/            # Model implementations
│   ├── evaluation/        # Evaluation metrics and visualization
│   └── utils/             # Utility functions
├── config/                 # Configuration files
├── models/                 # Trained model artifacts
├── results/               # Output results and reports
├── scripts/               # Executable scripts
├── tests/                 # Unit tests
└── docs/                  # Documentation
```

## 🔬 Methodology

### 1. Data Preprocessing
- Handle missing values
- Encode categorical variables
- Feature scaling and normalization
- Remove specified columns (education, native-country)
- Combine occupation categories

### 2. Model Training
- Implement 4 different algorithms
- Cross-validation for robust evaluation
- Hyperparameter optimization using grid search
- Model persistence and versioning

### 3. Evaluation
- Comprehensive metrics: Accuracy, Precision, Recall, F1-Score, ROC-AUC
- Confusion matrices and classification reports
- Feature importance analysis
- Model comparison and selection

## 📈 Results

Model performance comparison and detailed analysis available in:
- `results/reports/model_comparison.html`
- `results/metrics/validation_results.json`
- `notebooks/06_final_analysis.ipynb`

## 🛠️ Development

### Running Tests
```bash
pytest tests/
```

### Code Quality
```bash
# Format code
black src/ scripts/

# Linting
flake8 src/ scripts/
```

### Adding New Models
1. Create new model class in `src/models/`
2. Inherit from `BaseModel`
3. Implement required methods
4. Add configuration in `config/hyperparameters.yaml`
5. Update training script

## 📋 Requirements

- Python 3.8+
- scikit-learn 1.3.0+
- pandas 2.0.0+
- numpy 1.24.0+
- matplotlib 3.7.0+
- seaborn 0.12.0+
- See `requirements.txt` for complete list

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/new-feature`)
3. Commit changes (`git commit -am 'Add new feature'`)
4. Push to branch (`git push origin feature/new-feature`)
5. Create Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📧 Contact

**Author**: [Your Name]  
**Email**: [your.email@example.com]  
**GitHub**: [yourusername]

## 🙏 Acknowledgments

- UCI Machine Learning Repository for the Adult dataset
- scikit-learn community for excellent ML tools
- Open source contributors

---

*Built with ❤️ for machine learning education and research*
