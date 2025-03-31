# News Popularity Prediction

This repository contains the implementation of various machine learning models to predict online news popularity based on article features. The project compares different classification approaches to identify articles likely to receive high engagement (over 1,000 shares).

## Project Overview

Online news popularity is crucial for digital publishers, and the ability to predict which articles will gain traction can inform content strategy. This project explores several classification algorithms to predict whether an article will exceed 1,000 shares, using features from the UCI Online News Popularity dataset.

## Dataset

The analysis uses the [Online News Popularity dataset](https://archive.ics.uci.edu/ml/datasets/Online+News+Popularity) from UCI Machine Learning Repository, containing:
- ~40,000 articles
- 61 features including keyword metrics, timing features, and NLP-derived attributes
- The target variable: article shares (binarized to predict if shares > 1,000)

## Models Implemented

The project implements and compares the following models:

### Logistic Regression Variants
- Basic Logistic Regression
- Ridge Regression (L2 regularization)
- Lasso Regression (L1 regularization)
- Elastic Net Regression (Combined L1 and L2)

### K-Nearest Neighbors (KNN)
- Implementation with different k values (1-39)
- Evaluation based on accuracy, AUC, and cost metrics

### Random Forest
- Hyperparameter tuning for number of trees and variables per split
- Final model uses 1,000 trees and 12 variables per split

### Neural Networks
- Standard Neural Network
- Shallow Architecture (single hidden layer)
- Deep Architecture (multiple hidden layers)

## Evaluation Metrics

Models are evaluated using:
- Accuracy
- Precision, Recall, F1-Score
- AUC-ROC
- Custom cost functions (asymmetric costs for false positives and false negatives)

## Key Findings

The Random Forest model demonstrated the best overall performance with:
- Accuracy: 0.693
- Precision: 0.732
- Recall: 0.874
- F1-Score: 0.797
- Lowest total cost among all models tested

## Repository Structure

```
news-popularity-prediction/
├── setup.R               # Setup file with package installation and data loading
├── src/                   
│   ├── logistic_models.R   # Logistic regression variants
│   ├── knn_analysis.R      # KNN implementation
│   ├── random_forest.R     # Random Forest implementation
│   └── neural_networks.R   # Neural network models
└── README.md              # Project documentation
```

## Prerequisites

This project requires R with the following packages:
- glmnet
- pROC
- class
- rpart and rpart.plot
- randomForest
- caret
- PRROC
- knitr and kableExtra
- neuralnet
- h2o

## Usage

To run the analysis:

```r
# First, run the setup file to install packages and load data
source("setup.R")

# Then run individual model files as needed
source("src/logistic_models.R")
source("src/knn_analysis.R")
source("src/random_forest.R")
source("src/neural_networks.R")
```

## Future Work

Potential improvements include:
- Feature engineering to create more predictive variables
- Ensemble methods combining multiple model predictions
- Deep learning approaches with more complex architectures
- Time series analysis to account for temporal patterns in news popularity

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- UCI Machine Learning Repository for providing the dataset
- The original dataset creators: K. Fernandes, P. Vinagre and P. Cortez
