#####################################
#####  K-NN CLASSIFIER ##############
#####################################
set.seed(123)

# Function to calculate error rate
calculate_error_rate <- function(predictions.knn, true_labels) {
  error_rate.knn <- sum(predictions.knn != true_labels) / length(true_labels)
  return(error_rate.knn)
}

# Values of k to try
k_values <- c(1,3,5,7,9,11,13,15,17,19,21,23,25,27,29,31,33,35,37,39)

# Initialize results storage
results.knn <- data.frame(k = k_values, Error_Rate = numeric(length(k_values)), AUC = numeric(length(k_values)), Total_Costs = numeric(length(k_values)))

# Initialize storage for confusion matrices
confusion_matrices <- list()

# Iterate over k values
for (i in seq_along(k_values)) {
  k <- k_values[i]
  
  # Train K-NN model
  knn_model <- knn(train.data, valid.data, train.target, k = k)
  
  # Make predictions
  predictions.knn <- knn_model
  
  # Calculate error rate
  error_rate.knn <- calculate_error_rate(predictions.knn, valid.target)
  
  # Store results
  results.knn[i, "Error_Rate"] <- error_rate.knn
  
  # Confusion matrix
  confusion.knn <- table(valid.target, predictions.knn)
  confusion_matrices[[paste("k", k)]] <- confusion.knn
  print(confusion.knn)
  
  # Plot
  plot(confusion.knn, main = paste("Confusion Matrix (k =", k, ")"))
  
  # Calculate AUC
  roc_knn <- roc(valid.target, as.numeric(predictions.knn))
  auc_knn <- auc(roc_knn)
  print(paste("AUC for k =", k, ":", round(auc_knn, 3)))
  
  #store AUC in results
  results.knn[i, "AUC"] <- auc_knn
  
  # Plot ROC curve
  plot(roc_knn, main = paste("ROC Curve for k =", k))
  
  # Plot Precision-Recall curve
  pr_curve_knn <- pr.curve(valid.target, as.numeric(predictions.knn),curve = TRUE)
  plot(pr_curve_knn, main = paste("Precision-Recall Curve for k =", k))
  
  #calculate total costs
  total_cost_knn <- cost_fp * confusion.knn[1, 2] + cost_fn * confusion.knn[2, 1]
  
  #store Total Costs in results
  results.knn[i, "Total costs"] <- total_cost_knn
}

# Print results
print(results.knn)

# Find the value of k with the maximum AUC, minimum error and minimum cost
best_k_AUC <- results.knn[which.max(results.knn$AUC), "k"]
best_k_error <- results.knn[which.min(results.knn$Error_Rate), "k"]
best_k_costs <- results.knn[which.min(results.knn$`Total costs`), "k"]

# Print the best k values
print(paste("Best k based on Error Rate:", best_k_error))
print(paste("Best k based on AUC:", best_k_AUC))
print(paste("Best k based on Costs:", best_k_costs))

# Fit the final k-NN model on the training set with the best value of k
final_knn_model.1 <- knn(train.data, test.data, train.target, k = best_k_error)
final_knn_model.2 <- knn(train.data, test.data, train.target, k = best_k_AUC)
final_knn_model.3 <- knn(train.data, test.data, train.target, k = best_k_costs)

# Make predictions on the test set
predictions_knn_test.1 <- final_knn_model.1
predictions_knn_test.2 <- final_knn_model.2
predictions_knn_test.3 <- final_knn_model.3

# Error rate
error_rate_final_knn_model.1 <- calculate_error_rate(predictions_knn_test.1, test.target)
error_rate_final_knn_model.2 <- calculate_error_rate(predictions_knn_test.2, test.target)
error_rate_final_knn_model.3 <- calculate_error_rate(predictions_knn_test.3, test.target)

# Print error rates
cat("Error Rate for Final k-NN Model (Best k=35 based on Error Rate):", round(error_rate_final_knn_model.1, 3), "\n")
cat("Error Rate for Final k-NN Model (Best k=3 based on AUC):", round(error_rate_final_knn_model.2, 3), "\n")
cat("Error Rate for Final k-NN Model (Best k=3 based on AUC):", round(error_rate_final_knn_model.3, 3), "\n")

#evaluating model performance
#accuracy    
accuracy.1 <- sum(predictions_knn_test.1 == test.target) / length(test.target)
print(paste("Accuracy with k=35:", round(accuracy.1, 3)))     

accuracy.2 <- sum(predictions_knn_test.2 == test.target) / length(test.target)
print(paste("Accuracy with k=3:", round(accuracy.2, 3)))     

accuracy.3 <- sum(predictions_knn_test.3 == test.target) / length(test.target)
print(paste("Accuracy with k=3:", round(accuracy.3, 3)))     

#accuracy=0.686, 0.683 and 0.613, not that good. is dataframe not well balanced?

table(target)
# Calculate percentage of instances in the minority class
percentage_minority <- sum(target == 1) / nrow(data) * 100
print(paste("Percentage of Minority Class:", percentage_minority, "%"))
# Plot class distribution
barplot(table(target), main = "Class Distribution", col = c("red", "blue"), legend = TRUE)
# Calculate imbalance ratio
imbalance_ratio <- table(target)[[2]] / table(target)[[1]]
print(paste("Imbalance Ratio:", imbalance_ratio))

# yes, df is not balanced,using different metrics for evaluation

#precision, recall, F1-score
#confusion matrix 1
confusion.final_knn.1 <- table(test.target, predictions_knn_test.1)
confusion_matrices[[paste("k", k)]] <- confusion.final_knn.1
print(confusion.final_knn.1)

#metrics 1
true_positive.1 <- confusion.final_knn.1[2, 2]
precision.1 <- true_positive.1 / sum(confusion.final_knn.1[, 2])
recall.1 <- true_positive.1 / sum(confusion.final_knn.1[2, ])
f1_score.1 <- 2 * (precision.1 * recall.1) / (precision.1 + recall.1)

#print results 1
print(paste("Precision for k=35:", round(precision.1, 3)))
print(paste("Recall for k=35:", round(recall.1, 3)))
print(paste("F1 Score for k=35:", round(f1_score.1, 3)))

#confusion matrix 2
confusion.final_knn.2 <- table(test.target, predictions_knn_test.2)
confusion_matrices[[paste("k", k)]] <- confusion.final_knn.2
print(confusion.final_knn.2)

#metrics 2
true_positive.2 <- confusion.final_knn.2[2, 2]
precision.2 <- true_positive.2 / sum(confusion.final_knn.1[, 2])
recall.2 <- true_positive.2 / sum(confusion.final_knn.2[2, ])
f1_score.2 <- 2 * (precision.2 * recall.2) / (precision.2 + recall.2)

#print results 2
print(paste("Precision for k=3:", round(precision.2, 3)))
print(paste("Recall for k=3:", round(recall.2, 3)))
print(paste("F1 Score for k=3:", round(f1_score.2, 3)))

#confusion matrix 3
confusion.final_knn.3 <- table(test.target, predictions_knn_test.3)
confusion_matrices[[paste("k", k)]] <- confusion.final_knn.3
print(confusion.final_knn.3)

#metrics 1
true_positive.3 <- confusion.final_knn.3[2, 2]
precision.3 <- true_positive.3 / sum(confusion.final_knn.3[, 2])
recall.3 <- true_positive.3 / sum(confusion.final_knn.3[2, ])
f1_score.3 <- 2 * (precision.3 * recall.3) / (precision.3 + recall.3)

#print results 3
print(paste("Precision for k=39:", round(precision.3, 3)))
print(paste("Recall for k=39:", round(recall.3, 3)))
print(paste("F1 Score for k=39:", round(f1_score.3, 3)))

# best k-nn are k=3 and k=35, but costs-wise k=39 is to be preferred
