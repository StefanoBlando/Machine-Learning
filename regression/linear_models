## FULL LINEAR MODEL 

full_model <- lm(y_train~ ., data = X_train)
summary(full_model)
plot(full_model)

predictions_full_model <- predict(full_model, newdata = X_test)
MSE_full_model <- sqrt(1/nrow(X_test)*sum((y_test-predictions_full_model)^2))
summary(predictions_full_model)


## FORWARD STEPWISE SELECTION 

#model framework
min_model <- lm(y_train ~ 1, data = X_train)
max_model <-lm(y_train~ ., data= X_train)

# step forward model
forward_model_AIC <- step(min_model, direction='forward', scope=formula(max_model), trace=0)
forward_model_AIC$anova
forward_model_AIC$coefficients
summary(forward_model_AIC)
#predictions
predictions_forward_model_AIC <- predict(forward_model_AIC, newdata= X_test,type="response")
MSE_forward_model_AIC <- sqrt(1/nrow(X_test)*sum((y_test-predictions_forward_model_AIC)^2))
summary(predictions_forward_model_AIC)

# stepBIC forward model
n=nrow(X_train)
forward_model_BIC <- stepAIC(min_model, direction='forward', scope=formula(max_model),k=log(n), trace=0)
forward_model_BIC$anova
forward_model_BIC$coefficients
summary(forward_model_BIC)

#predictions
predictions_forward_model_BIC <- predict(forward_model_BIC, newdata = X_test, type="response")
MSE_forward_model_BIC <- sqrt(1/nrow(X_test)*sum((y_test-predictions_forward_model_BIC)^2))
summary(predictions_forward_model_BIC)
