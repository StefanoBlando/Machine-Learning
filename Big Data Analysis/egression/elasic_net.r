## ELASTIC NET REGRESSION
# base model
elasticnet_model = glmnet(x=x, y=y, family = "poisson", alpha=0.5)
plot(elasticnet_model$lambda,pch=17,col="firebrick4", type = "b", main = "Penalty Grid")
dim(elasticnet_model$beta)
image(elasticnet_model$beta)
matplot(t(elasticnet_model$beta), type = "l")

lgrid = elasticnet_model$lambda
agrid = seq(0,1,length.out=25)

# cross validation
set.seed(123)
cvloop = cv.glmnet(x=x, y=y, family="poisson", alpha=agrid[1], lambda = lgrid)
cvloop = cbind(rep(agrid[1],length(cvloop$lambda)),cvloop$lambda,cvloop$cvm)
for(i in 2:length(agrid)){
  res_i =   cv.glmnet(x=x, y=y, family="poisson", alpha=agrid[i], lambda = lgrid)
  cvloop = rbind(cvloop, cbind(rep(agrid[i],length(res_i$lambda)),res_i$lambda,res_i$cvm))
}
cvloop = as.data.frame(cvloop)
names(cvloop) <- c("alpha","lambda","cvm")

ggplot(cvloop, aes(x=lambda, y=cvm, group=alpha, color=alpha))+geom_line()

alphastar = cvloop[which.min(cvloop$cvm),1]
lambdastar = cvloop[which.min(cvloop$cvm),2]

# best selected model
elasticnet_model_cv = glmnet(x=x, y=y, family = "poisson", alpha = alphastar, lambda = lambdastar)

round(cbind(rbind(lasso_model_cv$a0, lasso_model_cv$beta),rbind(ridge_model_cv$a0, ridge_model_cv$beta),rbind(elasticnet_model_cv$a0,elasticnet_model_cv$beta)),5)

# predictions and visualization
predictions_elasticnet = predict(elasticnet_model_cv, newx = as.matrix(X_test))
plot(y_test, predictions_elasticnet, pch=19)
abline(b=1, a=0, col=2, lwd=2)

#MSE
MSE_elasticnet <-sqrt(1/nrow(X_test)*sum((y_test-predictions_elasticnet)^2))

