# Big Data Analysis for Economics and Finance

This repository contains multiple data analysis projects focused on applying big data techniques to economic, financial, and political data. Each project demonstrates different methodologies and approaches for extracting insights from large datasets.

## Projects

### 1. Classification: News Popularity Prediction

A machine learning project that implements various classification algorithms to predict whether online news articles will become popular (receive more than 1,000 shares).

**Techniques explored:**
- Logistic Regression with regularization (Ridge, Lasso, Elastic Net)
- K-Nearest Neighbors
- Random Forest
- Neural Networks

[Go to project →](./classification/)

### 2. Regression: News Popularity Analysis

A comprehensive regression analysis that explores the relationship between article features and sharing behavior using linear and regularized models to predict the exact number of shares.

**Techniques explored:**
- Full Linear Models
- Stepwise Feature Selection
- Ridge Regression
- Lasso Regression
- Elastic Net Regression

[Go to project →](./regression/)

### 3. Clustering: Voting Patterns Analysis

An advanced clustering analysis that identifies patterns and similarities in republican voting behavior across different countries over time.

**Techniques explored:**
- Principal Component Analysis (PCA)
- K-means Clustering
- Trimmed K-means (Robust Clustering)
- Cluster Validation Methods
- Cluster Trimmed Likelihood Curves

[Go to project →](./clustering/)

## Repository Structure

```
big-data-analysis/
├── classification/                 # News popularity classification project directory
│   ├── README.md                   # Classification project documentation
│   ├── data_processing.R           # Data preprocessing for classification
│   ├── knn.R                       # K-Nearest Neighbors implementation
│   ├── logistic_regression.R       # Logistic regression variants
│   ├── neural_networks.R           # Neural networks implementation
│   └── random_forest.R             # Random Forest implementation
│
├── regression/                     # News popularity regression project directory
│   ├── README.md                   # Regression project documentation
│   ├── data_processing.R           # Data preprocessing for regression
│   ├── elasic_net.R                # Elastic Net regression
│   ├── lasso.R                     # Lasso regression
│   ├── linear_models.R             # Linear models and stepwise selection
│   ├── results.R                   # Results analysis
│   └── ridge.R                     # Ridge regression
│
├── clustering/                     # Voting patterns clustering project directory
│   ├── README.md                   # Clustering project documentation
│   └── clustering.R                # Complete clustering analysis
│
├── LICENSE                         # License file
└── README.md                       # Main repository documentation
```

## Datasets

The projects use the following datasets:

- **News projects**: [Online News Popularity dataset](https://archive.ics.uci.edu/ml/datasets/Online+News+Popularity) from UCI Machine Learning Repository
- **Voting patterns project**: Republican voting dataset with countries and yearly voting percentages

## Prerequisites

These projects require R with various packages. Each project's data_processing.R file handles the installation and loading of required packages.

## Running the Projects

Each project has its own README file with specific instructions. Generally:

1. Clone this repository
2. Navigate to the project directory of interest
3. Run the data_processing.R file first to install dependencies and load data
4. Run the individual analysis files

## Key Findings

### Classification Project
The Random Forest model demonstrated superior performance in classifying popular articles, with an accuracy of 0.693, precision of 0.732, recall of 0.874, and F1-score of 0.797.

### Regression Project
The Lasso regression model provided the best performance in predicting the exact number of shares. Global sentiment polarity and the rate of negative words were identified as significant predictors of article popularity.

### Clustering Project
Two distinct clusters of countries emerged based on voting patterns, with PCA revealing significant temporal patterns in the data. Trimmed K-means provided more robust clustering by effectively handling outliers.

## License

This repository is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- UCI Machine Learning Repository for providing the news popularity dataset
- The dataset providers for the republican voting data
- The original dataset creators and R package developers
