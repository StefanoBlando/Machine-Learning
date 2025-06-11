## CONCLUSIONS

MSE_values <- c(MSE_full_model, MSE_forward_model_AIC, MSE_forward_model_BIC, MSE_ridge, MSE_lasso, MSE_elasticnet)
MSE_names <- c("MSE_full_model", "MSE_forward_model_AIC", "MSE_forward_model_BIC", "MSE_ridge", "MSE_lasso", "MSE_elasticnet")

min_index <- which.min(MSE_values)
min_name <- MSE_names[min_index]
min_value <- MSE_values[min_index]

cat("Minimum MSE:", min_name, "(", min_value, ")\n")


The best model is LASSO.

The effect of global sentiment polarity seems to be influencing the popularity of online news, but said effect 
has to be calibrated by the rate of positive or negative words of the article, with a correlation 
that tends to prefer neative words. 
