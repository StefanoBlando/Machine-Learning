#####################################
#####  RANDOM FOREST ################
#####################################

set.seed(123)

# Define the range for the number of trees
ntree_range <- seq(500, 2000, by = 250)

# Initialize results storage for the number of trees
results_rf_ntree <- data.frame(ntree = numeric(length(ntree_range)), AUC = numeric(length(ntree_range)))

# Hyperparameter tuning for the number of trees
for (i in seq_along(ntree_range)) {
  # Fit Random Forest model on the training set
  rf_model <- randomForest(as.factor(train.target) ~ ., data = train.data, ntree = ntree_range[i])
  
  # Make predictions on the validation set
  predictions_rf_val <- predict(rf_model, newdata = valid.data)
  
  # Evaluate Random Forest model on the validation set
  roc_rf_val <- roc(valid.target, as.numeric(predictions_rf_val))
  auc_rf_val <- auc(roc_rf_val)
  
  # Store results
  results_rf_ntree[i, "ntree"] <- ntree_range[i]
  results_rf_ntree[i, "AUC"] <- auc_rf_val
}

# Print results
print(results_rf_ntree)

# Choose the optimal number of trees (e.g., the one with the highest AUC)
best_ntree <- results_rf_ntree[which.max(results_rf_ntree$AUC), "ntree"]

# Define the range for the number of variables at each split
mtry_range <- seq(1, ncol(train.data)-1, by = 1)

# Initialize results storage for the number of variables at each split
results_rf_mtry <- data.frame(mtry = numeric(length(mtry_range)), AUC = numeric(length(mtry_range)))

# Hyperparameter tuning for the number of variables at each split
for (j in seq_along(mtry_range)) {
  # Fit Random Forest model on the training set
  rf_model <- randomForest(as.factor(train.target) ~ ., data = train.data, ntree = best_ntree, mtry = mtry_range[j])
  
  # Make predictions on the validation set
  predictions_rf_val <- predict(rf_model, newdata = valid.data)
  
  # Evaluate Random Forest model on the validation set
  roc_rf_val <- roc(valid.target, as.numeric(predictions_rf_val))
  auc_rf_val <- auc(roc_rf_val)
  
  # Store results
  results_rf_mtry[j, "mtry"] <- mtry_range[j]
  results_rf_mtry[j, "AUC"] <- auc_rf_val
}

# Print results
print(results_rf_mtry)

# Choose the optimal number of variables at each split (the one with the highest AUC)
best_mtry <- results_rf_mtry[which.max(results_rf_mtry$AUC), "mtry"]

# Train the final Random Forest model on the training set with the optimal hyperparameters
final_rf_model <- randomForest(as.factor(train.target) ~ ., data = train.data, ntree = 1000, mtry = 12)

# Make predictions on the test set
predictions_rf_test <- predict(final_rf_model, newdata = test.data)

# Evaluate Random Forest model on the test set
roc_rf_test <- roc(test.target, as.numeric(predictions_rf_test))
auc_rf_test <- auc(roc_rf_test)

# Print final AUC on the test set
print(paste("Final AUC on Test Set:", round(auc_rf_test, 3)))

#confusion matrix rf
confusion.final_rf <- table(test.target, predictions_rf_test)
confusion_matrices[[paste("k", k)]] <- confusion.final_rf
print(confusion.final_rf)

#total costs rf
total_cost_rf <- cost_fp * confusion.final_rf[1, 2] + cost_fn * confusion.final_rf[2, 1]

print(total_cost_rf)

#metrics 
accuracy.rf <- sum(predictions_rf_test == test.target) / length(test.target)
print(paste("Accuracy with trees=1000 and variables=12:", round(accuracy.rf, 3)))     

true_positive.rf <- confusion.final_rf[2, 2]
precision.rf <- true_positive.rf / sum(confusion.final_rf[, 2])
recall.rf <- true_positive.rf / sum(confusion.final_rf[2, ])
f1_score.rf <- 2 * (precision.rf * recall.rf) / (precision.rf + recall.rf)

#print results 3
print(paste("Precision for rf:", round(precision.rf, 3)))
print(paste("Recall for rf:", round(recall.rf, 3)))
print(paste("F1 Score for rf:", round(f1_score.rf, 3)))
