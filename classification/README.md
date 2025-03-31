# News Popularity Regression Analysis

This repository contains a comprehensive regression analysis of online news popularity using various linear and regularized models. The project aims to predict article shares and identify key features that drive content engagement.

## Project Overview

Understanding what drives news article popularity is crucial for digital publishers. This project analyzes the relationship between article features and sharing behavior using different regression techniques to identify the most effective prediction model and the most influential features affecting news popularity.

## Dataset

The analysis uses the [Online News Popularity dataset](https://archive.ics.uci.edu/ml/datasets/Online+News+Popularity) from UCI Machine Learning Repository, containing:
- ~40,000 articles
- 61 features including keyword metrics, timing features, and NLP-derived attributes
- The target variable: article shares

## Models Implemented

The project implements and compares the following regression models:

### Traditional Regression Approaches
- Full Linear Model with all predictors
- Forward Stepwise Selection with AIC criterion
- Forward Stepwise Selection with BIC criterion

### Regularized Regression Methods
- Ridge Regression
- Lasso Regression
- Elastic Net Regression

## Evaluation Metrics

Models are evaluated using:
- Root Mean Squared Error (RMSE)
- Model coefficient analysis
- Prediction visualization

## Key Findings

- The Lasso regression model demonstrated the best performance with the lowest RMSE
- Global sentiment polarity appears to influence article popularity
- The effect of sentiment is moderated by the rate of positive/negative words
- Negative sentiment words show stronger correlation with article shares

## Repository Structure

```
news-popularity-regression/
├── setup.R                 # Setup file with package installation and data loading
├── src/                   
│   ├── data_processing.R   # Data cleaning and preparation
│   ├── linear_models.R     # Full model and stepwise selection
│   ├── ridge_model.R       # Ridge regression implementation
│   ├── lasso_model.R       # Lasso regression implementation
│   └── elastic_net.R       # Elastic Net implementation
└── README.md              # Project documentation
```

## Prerequisites

This project requires R with the following packages:
- glmnet
- MASS
- SIS
- caret
- ggplot2
- car
- dplyr

## Usage

To run the analysis:

```r
# First, run the setup file to install packages and load data
source("setup.R")

# Then run individual model files as needed
source("src/data_processing.R")
source("src/linear_models.R")
source("src/ridge_model.R")
source("src/lasso_model.R")
source("src/elastic_net.R")
```

## Future Work

Potential improvements include:
- Feature engineering to create more predictive variables
- Non-linear regression models (e.g., GAMs, polynomial regression)
- Time series analysis to account for temporal patterns in news popularity
- Ensemble methods combining multiple model predictions

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- UCI Machine Learning Repository for providing the dataset
- The original dataset creators: K. Fernandes, P. Vinagre and P. Cortez
