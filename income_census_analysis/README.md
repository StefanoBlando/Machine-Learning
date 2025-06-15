# Income Census Analysis

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.3.0-orange)](https://scikit-learn.org/)
[![License](https://img.shields.io/badge/License-AGPL%20v3-blue.svg)](https://www.gnu.org/licenses/agpl-3.0)

A comprehensive machine learning project for binary classification to predict whether an individual's income exceeds $50K based on demographic and employment data from the Adult Census dataset.

## 🎯 Project Objective

This project implements and compares four machine learning algorithms to predict income levels:
- **Random Forest Classifier**
- **Gradient Boosting Classifier**  
- **Support Vector Machine (SVM)**
- **Logistic Regression**

The goal is to identify the best performing model through rigorous hyperparameter optimization and cross-validation.

## 📊 Dataset Overview

The Adult Census dataset contains demographic information with the following characteristics:

- **Target Variable**: Income level (>50K, <=50K)
- **Features**: 13 demographic and employment attributes
- **Size**: 32,561 samples
- **Task**: Binary classification

### Key Preprocessing Steps:
- Removal of `education` and `native-country` features
- Occupation consolidation into 5 main categories
- Train/validation split: 70%/30% (random_state=123)

## 🚀 Quick Start

### Prerequisites
```bash
Python 3.8+
pip or conda package manager
```

### Installation
```bash
# Clone the repository
git clone https://github.com/yourusername/income_census_analysis.git
cd income_census_analysis

# Install dependencies
pip install -r requirements.txt

# Alternative: conda environment
conda env create -f environment.yml
conda activate income_census
```

### Usage
```bash
# Run complete analysis pipeline
python scripts/train_models.py

# Run hyperparameter tuning
python scripts/hyperparameter_tuning.py

# Generate evaluation report
python scripts/evaluate_models.py

# View results in Jupyter notebooks
jupyter notebook notebooks/
```

## 📁 Project Structure

```
income_census_analysis/
├── data/
│   ├── raw/                    # Original dataset
│   ├── processed/              # Cleaned and engineered data
│   └── external/               # External data sources
├── src/
│   ├── data/                   # Data processing modules
│   ├── models/                 # ML model implementations
│   ├── evaluation/             # Model evaluation utilities
│   └── utils/                  # Helper functions
├── notebooks/                  # Jupyter notebooks for analysis
├── config/                     # Configuration files
├── models/                     # Trained model artifacts
├── results/                    # Output files and reports
├── scripts/                    # Executable scripts
└── tests/                      # Unit tests
```

## 🔧 Technical Implementation

### Data Processing Pipeline
1. **Data Loading**: Import and initial exploration
2. **Feature Engineering**: Create meaningful features from raw data
3. **Data Cleaning**: Handle missing values and outliers
4. **Feature Selection**: Remove specified features (education, native-country)
5. **Categorical Encoding**: Transform categorical variables
6. **Scaling**: Normalize numerical features
7. **Train/Test Split**: 70/30 split with stratification

### Model Implementation
Each model is implemented with:
- Custom hyperparameter grids
- Cross-validation for robust evaluation
- Feature importance analysis
- Performance metrics tracking

### Evaluation Metrics
- **Accuracy**: Overall classification accuracy
- **Precision**: Positive predictive value
- **Recall**: Sensitivity/True positive rate
- **F1-Score**: Harmonic mean of precision and recall
- **ROC-AUC**: Area under the ROC curve
- **Confusion Matrix**: Detailed classification results

## 📈 Results Summary

The best performing models will be documented in `results/reports/model_comparison.html` with:
- Comparative performance metrics
- Feature importance rankings
- Hyperparameter optimization results
- Cross-validation scores
- Statistical significance tests

## 🛠️ Development

### Running Tests
```bash
# Run all tests
pytest tests/

# Run specific test file
pytest tests/test_models.py

# Run with coverage
pytest --cov=src tests/
```

### Code Quality
```bash
# Format code
black src/ scripts/ tests/

# Lint code
flake8 src/ scripts/ tests/

# Type checking
mypy src/
```

## 📚 Documentation

Detailed documentation is available in the `docs/` directory:
- [Project Overview](docs/project_overview.md)
- [Data Description](docs/data_description.md)
- [Methodology](docs/methodology.md)
- [API Reference](docs/api_reference.md)

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the GNU Affero General Public License v3.0 (AGPL-3.0) - see the [LICENSE](LICENSE) file for details.


## 🙏 Acknowledgments

- UCI Machine Learning Repository for the Adult Census dataset
- Scikit-learn community for excellent ML tools
- Open source contributors who made this project possible

## 📞 Contact

For questions or collaboration opportunities:
- Email: your.email@example.com
- LinkedIn: [Your LinkedIn](https://linkedin.com/in/yourprofile)
- GitHub: [@yourusername](https://github.com/yourusername)

---

⭐ **Star this repository if you found it helpful!**
