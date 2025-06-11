## RIDGE REGRESSION

# input processing
y = y_train
x = as.matrix(X_train)

# base model
ridge_model = glmnet(x=x, y=y, alpha = 0, family = "poisson", nlambda = 250)
plot(ridge_model$lambda,pch=17,col="firebrick4", type = "b", main = "Penalty Grid")
dim(ridge_model$beta)
image(ridge_model$beta)
matplot(t(ridge_model$beta), type = "l")

#cross-validation
set.seed(123)
cv_ridge = cv.glmnet(x=x, y=y, alpha=0, family="poisson", nlambda=250)
plot(log(cv_ridge$lambda), cv_ridge$cvm,type = "b", pch=19)
plot(cv_ridge)

mridge = which.min(cv_ridge$cvm) 

plot(log(cv_ridge$lambda)[(mridge-10):(mridge+10)], cv_ridge$cvm[(mridge-10):(mridge+10)],type = "b", pch=19)
abline(v=log(cv_ridge$lambda[mridge]), lty=4)  

# best selected model
ridge_model_cv = glmnet(x=x, y=y, family = "poisson", alpha=0,lambda = cv_ridge$lambda[mridge])

round(cbind(rbind(ridge_model_cv$a0, ridge_model_cv$beta)[-which(is.na(coef(full_model))),],coef(full_model)[-which(is.na(coef(full_model)))]),5)
predictions_ridge = predict(ridge_model_cv, newx = as.matrix(X_test), type="response")

# predictions and visualization
plot(y_test, predictions_ridge, pch=19)
abline(b=1, a=0, col=2, lwd=2)

ggplot(X_test, aes(x=predictions_ridge, y=y_test)) + 
  geom_point() +
  geom_abline(intercept=0, slope=1) +
  labs(x='Predicted Values', y='Actual Values', title='Predicted vs. Actual Values')

#MSE
MSE_ridge <- sqrt(1/nrow(X_test)*sum((y_test-predictions_ridge)^2))
