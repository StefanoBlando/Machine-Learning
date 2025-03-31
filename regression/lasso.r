## LASSO REGRESSION

# base model
lasso_model <- glmnet(x=x, y=y, alpha = 1, family = "poisson", nlambda = 250)
plot(lasso_model$lambda,pch=17,col="firebrick4", type = "b", main = "Penalty Grid")
dim(lasso_model$beta)
image(lasso_model$beta)
matplot(t(lasso_model$beta), type = "l")

# cross validation
set.seed(123)
cv_lasso = cv.glmnet(x=x, y=y, family="poisson", nlambda=250, alpha=1)
plot(log(cv_lasso$lambda), cv_lasso$cvm,type = "b", pch=19)
plot(cv_lasso)

mlasso = which.min(cv_lasso$cvm)

plot(log(cv_lasso$lambda)[(mlasso-10):(mlasso+10)], cv_lasso$cvm[(mlasso-10):(mlasso+10)],type = "b", pch=19)
abline(v=log(cv_lasso$lambda[mlasso]), lty=4) 

# best selected model 
lasso_model_cv = glmnet(x=x, y=y, family = "poisson", alpha=1,lambda = cv_lasso$lambda[mlasso])
round(cbind(rbind(lasso_model_cv$a0, lasso_model_cv$beta),rbind(ridge_model_cv$a0, ridge_model_cv$beta)),5)

# predictions and visualization
predictions_lasso = predict(lasso_model_cv, newx = as.matrix(X_test),type="response")
plot(y_test, predictions_lasso, pch=19)
abline(b=1, a=0, col=2, lwd=2)

#MSE
MSE_lasso <-sqrt(1/nrow(X_test)*sum((y_test-predictions_lasso)^2))
