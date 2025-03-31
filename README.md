# Big Data Analysis 

This repository contains multiple data analysis projects focused on applying big data techniques to economic and financial data. Each project demonstrates different methodologies and approaches for extracting insights from large datasets.

## Projects

### 1. News Popularity Classification

A machine learning project that implements various classification algorithms to predict whether online news articles will become popular (receive more than 1,000 shares).

**Techniques explored:**
- Logistic Regression with regularization (Ridge, Lasso, Elastic Net)
- K-Nearest Neighbors
- Random Forest
- Neural Networks

[Go to project →](./news-popularity-classification/)

### 2. News Popularity Regression Analysis

A comprehensive regression analysis that explores the relationship between article features and sharing behavior using linear and regularized models to predict the exact number of shares.

**Techniques explored:**
- Full Linear Models
- Stepwise Feature Selection
- Ridge Regression
- Lasso Regression
- Elastic Net Regression

[Go to project →](./news-popularity-regression/)

## Repository Structure

```
big-data-analysis/
├── news-popularity-classification/    # Classification project directory
│   ├── setup.R                        # Setup file for classification project
│   ├── src/                           # Classification models source code
│   │   ├── logistic_models.R
│   │   ├── knn_analysis.R
│   │   ├── random_forest.R
│   │   └── neural_networks.R
│   └── README.md                      # Classification project documentation
│
├── news-popularity-regression/        # Regression project directory
│   ├── setup.R                        # Setup file for regression project
│   ├── src/                           # Regression models source code
│   │   ├── data_processing.R
│   │   ├── linear_models.R
│   │   ├── ridge_model.R
│   │   ├── lasso_model.R
│   │   └── elastic_net.R
│   └── README.md                      # Regression project documentation
│
├── data/                              # Shared data directory
│   └── OnlineNewsPopularity.csv       # Dataset used by both projects
│
└── README.md                          # Main repository documentation
```

## Dataset

Both projects use the [Online News Popularity dataset](https://archive.ics.uci.edu/ml/datasets/Online+News+Popularity) from UCI Machine Learning Repository, which contains:
- ~40,000 articles
- 61 features including keyword metrics, timing features, and NLP-derived attributes
- The target variable: article shares

## Prerequisites

These projects require R with various packages. Each project's setup.R file handles the installation and loading of required packages.

## Running the Projects

Each project has its own README file with specific instructions. Generally:

1. Clone this repository
2. Navigate to the project directory of interest
3. Run the setup.R file first to install dependencies and load data
4. Run the individual analysis files

## Key Findings

### Classification Project
The Random Forest model demonstrated superior performance in classifying popular articles, with an accuracy of 0.693, precision of 0.732, recall of 0.874, and F1-score of 0.797.

### Regression Project
The Lasso regression model provided the best performance in predicting the exact number of shares. Global sentiment polarity and the rate of negative words were identified as significant predictors of article popularity.

## License

This repository is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- UCI Machine Learning Repository for providing the dataset
- The original dataset creators: K. Fernandes, P. Vinagre and P. Cortez
