#####################################
##### LOGISTIC REGRESSION ###########
#####################################
#base model
logistic.model <- glm(train.target ~ ., data = train.data , family="binomial")

#data processing
dmx.train <- as.matrix(train.data)
dmx.valid <- as.matrix(valid.data)
dmx.test <- as.matrix(test.data)

dmy.train <- as.matrix(train.target)
dmy.valid <- as.matrix(valid.data)
dmy.test <- as.matrix(test.target)

#ridge regression
out.ridge = glmnet(x=train.data, y=train.target, alpha = 0, family = "binomial", nlambda = 250)
plot(out.ridge$lambda,pch=17,col="firebrick4", type = "b", main = "Penalty Grid")
dim(out.ridge$beta)
image(out.ridge$beta)
matplot(t(out.ridge$beta), type = "l")

#select lambda and find best cv ridge
set.seed(123)
cv.logistic.ridge.model <- cv.glmnet(dmx.train, dmy.train, family = "binomial", nlambda=250, alpha=0, standardize=TRUE)
ridge = glmnet(x=train.data, y=train.target, family="binomial", alpha=0, lambda = cv.logistic.ridge.model$lambda.min)

# lasso 
out.lasso = glmnet(x=train.data, y=train.target, alpha = 1, family = "binomial", nlambda = 250)
plot(out.lasso$lambda,pch=17,col="firebrick4", type = "b", main = "Penalty Grid")
dim(out.lasso$beta)
image(out.lasso$beta)
matplot(t(out.lasso$beta), type = "l")

#select lambda and find best cv lasso
set.seed(789)
cv.logistic.lasso.model <- cv.glmnet(dmx.train, dmy.train, family = "binomial", alpha = 1,standardize=TRUE)
lasso = glmnet(x=train.data, y=train.target, family="binomial", alpha=1, lambda = cv.logistic.lasso.model$lambda.min)

# elastic net 
out.en = glmnet(x=train.data, y=train.target, alpha = 0.5, family = "binomial", nlambda = 250)
plot(out.en$lambda,pch=17,col="firebrick4", type = "b", main = "Penalty Grid")
dim(out.en$beta)
image(out.en$beta)
matplot(t(out.en$beta), type = "l")

#select lambda and find best cv en
set.seed(123)
cv.logistic.en.model <- cv.glmnet(dmx.train, dmy.train, family = "binomial", alpha = 0.5, standardize=TRUE)

lgrid = out.en$lambda
agrid = seq(0,1,length.out=25)

set.seed(123)
cvloop = cv.glmnet(dmx.train, dmy.train, family = "binomial", alpha=agrid[1], lambda = lgrid)
cvloop = cbind(rep(agrid[1],length(cvloop$lambda)),cvloop$lambda,cvloop$cvm)
for(i in 2:length(agrid)){
  res_i =   cv.glmnet(dmx.train, dmy.train, family = "binomial", alpha=agrid[i], lambda = lgrid)
  cvloop = rbind(cvloop, cbind(rep(agrid[i],length(res_i$lambda)),res_i$lambda,res_i$cvm))
}
cvloop = as.data.frame(cvloop)
names(cvloop) <- c("alpha","lambda","cvm")
library(ggplot2)
ggplot(cvloop, aes(x=lambda, y=cvm, group=alpha, color=alpha))+geom_line()
cvloop[which.min(cvloop$cvm),]
alphastar = cvloop[which.min(cvloop$cvm),1]
lambdastar = cvloop[which.min(cvloop$cvm),2]

en = glmnet(dmx.train, dmy.train, family = "binomial", alpha = alphastar, lambda = lambdastar)

# Cross-validation results for logistic models
print(logistic.model)
print(ridge)
print(lasso)
print(en)

#forecasts
pr.logistic.model <- predict(logistic.model, newdata=test.data, type="response")
pr.ridge <- predict.glmnet(ridge, newx= dmx.test, type="response")
pr.lasso <- predict.glmnet(lasso, newx= dmx.test, type="response")
pr.en <- predict.glmnet(en, newx= dmx.test, type="response")

# Assuming threshold is 0.5
threshold <- 0.5

# Convert predictions to binary class labels
predicted_labels_logistic <- as.integer(pr.logistic.model > threshold)
predicted_labels_ridge <- as.integer(pr.ridge > threshold)
predicted_labels_lasso <- as.integer(pr.lasso > threshold)
predicted_labels_en <- as.integer(pr.en > threshold)

# Calculating ROC and AUC
roc.logistic.model <- roc(test.target, predicted_labels_logistic)
roc.ridge <- roc(test.target, predicted_labels_ridge)
roc.lasso <- roc(test.target, predicted_labels_lasso)
roc.en <- roc(test.target, predicted_labels_en)

auc_logistic_model <- auc(roc(test.target, predicted_labels_logistic))
auc_ridge <- auc(roc(test.target, predicted_labels_ridge))
auc_lasso<- auc(roc(test.target, predicted_labels_lasso))
auc_en <- auc(roc(test.target, predicted_labels_en))

# Print AUC 
print(c(
  Logistic_Model = auc_logistic_model,
  CV_Ridge_Model = auc_ridge,
  CV_Lasso_Model = auc_lasso,
  CV_Elastic_Net_Model = auc_en
))

# Plot ROC curves
plot(roc.logistic.model, col = "blue", main = "ROC Curves for Logistic Regression Models")
plot(roc.ridge, col = "green", add = TRUE)
plot(roc.lasso, col = "orange", add = TRUE)
plot(roc.en, col = "purple", add = TRUE)

legend("bottomright", legend = c("Logistic Model", "CV Ridge Model", "CV Lasso Model", "CV Elastic Net Model"),
       col = c("blue", "green", "orange", "purple"), lty = 1)

text(0.8, 0.2, paste("AUC Logistic Model =", round(auc(roc.logistic.model), 3)), col = "blue")
text(0.8, 0.15, paste("AUC CV Ridge Model =", round(auc(roc.ridge), 3)), col = "green")
text(0.8, 0.1, paste("AUC CV Lasso Model =", round(auc(roc.lasso), 3)), col = "orange")
text(0.8, 0.05, paste("AUC CV Elastic Net Model =", round(auc(roc.en), 3)), col = "purple")

# AUC below 0.7, not a good result

# Confusion matrix 
confusion_logistic_model <- confusionMatrix(factor(round(predicted_labels_logistic)), factor(test.target), positive="1")
confusion_cv_logistic_ridge_model <- confusionMatrix(factor(round(predicted_labels_ridge)), factor(test.target),positive="1")
confusion_cv_logistic_lasso_model <- confusionMatrix(factor(round(predicted_labels_lasso)), factor(test.target),positive="1")
confusion_cv_logistic_en_model <- confusionMatrix(factor(round(predicted_labels_en)), factor(test.target),positive="1")

# Print confusion matrices
print(list(
  Logistic_Model = confusion_logistic_model,
  CV_Ridge_Model = confusion_cv_logistic_ridge_model,
  CV_Lasso_Model = confusion_cv_logistic_lasso_model,
  CV_Elastic_Net_Model = confusion_cv_logistic_en_model
))

# Define false positive and false negative costs
cost_fp <- 1  # Cost of false positive
cost_fn <- 10  # Cost of false negative

# Function to calculate total cost
calculate_total_cost <- function(predictions, true_labels, threshold) {
  predicted_labels <- ifelse(predictions > threshold, 1, 0)
  confusion_matrix <- confusionMatrix(factor(predicted_labels), factor(true_labels))
  total_cost <- cost_fp * confusion_matrix$table[1, 2] + cost_fn * confusion_matrix$table[2, 1]
  return(total_cost)
}

# Logistic Regression
total_cost_logistic_model <- calculate_total_cost(pr.logistic.model, test.target, threshold)
print(paste("Total Cost for Logistic Regression Model at Threshold 0.5:", total_cost_logistic_model))

# Ridge Regression
total_cost_ridge <- calculate_total_cost(pr.ridge, test.target, threshold)
print(paste("Total Cost for Ridge Model at Threshold 0.5:", total_cost_ridge))

# Lasso Regression
total_cost_lasso <- calculate_total_cost(pr.lasso, test.target, threshold)
print(paste("Total Cost for Lasso Model at Threshold 0.5:", total_cost_lasso))

# Elastic Net
total_cost_en <- calculate_total_cost(pr.en, test.target, threshold)
print(paste("Total Cost for Elastic Net Model at Threshold 0.5:", total_cost_en))

# Create a data frame with the results
results <- data.frame(
  Model = c("Logistic Regression", "Ridge Regression", "Lasso Regression", "Elastic Net"),
  Total_Cost = c(total_cost_logistic_model, total_cost_ridge, total_cost_lasso, total_cost_en)
)

# Print the table
kable(results, format = "html", caption = "Total Costs for Different Models at Threshold 0.5") %>%
  kable_styling()

#ridge has the minimum cost (almost equal to elastic net)
