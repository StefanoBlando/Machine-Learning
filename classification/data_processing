#Libraries 
library(glmnet)
library(pROC)
library(class)
library(rpart)
library(rpart.plot)
library(randomForest)
library(caret)
library(PRROC)
library(knitr)
library(kableExtra)
library(neuralnet)
library(h2o)

#loading data
data <- read.csv("C/data/OnlineNewsPopularity.csv")

# Select candidate predictors
predictors <- data[, c(3, 4, 8, 10:19, 39, 45:48)]

# Define the target variable
target <- ifelse(data$shares > 1000, 1, 0)

# Split the data into training, validation, and testing sets
set.seed(123)
train.index <- sample(1:nrow(data), 0.7 * nrow(data))
valid.index <- sample(setdiff(1:nrow(data), train.index), 0.2 * nrow(data))
test.index <- setdiff(1:nrow(data), union(train.index, valid.index))

train.data <- predictors[train.index, ]
train.target <- target[train.index]
valid.data <- predictors[valid.index, ]
valid.target <- target[valid.index]
test.data <- predictors[test.index, ]
test.target <- target[test.index]
