#####################################
#####  NEURAL NETWORK  ##############
#####################################

# prepare data for the model
x          <- data[, unlist(lapply(data, is.numeric))]
x          <- scale(x) # recommended with networks 
y          <- data[, 61]
test      <- sample(1:nrow(data)*0.1)
train      <- -test 
x_train    <- x[train,c(3, 4, 8, 10:19, 39, 45:48)]
x_test     <- x[test,]
y_train    <- y[train]
y_test     <- y[test]
df_train   <- data.frame(x_train, y_train)
df_test    <- data.frame(x_test, y_test)
model_list <- "y_train ~ ." 

# build model
hidden    <- c(15,9,6,3)
threshold <- 1.5
stepmax   <- 5000 
rep       <- 3
model     <- neuralnet(model_list, data = df_train, hidden = hidden, threshold = threshold,
                       stepmax = stepmax, rep = rep, lifesign = "minimal", 
                       act.fct = "logistic", linear.output = FALSE)
plot(model)

# look at predictions and confusion matrix
thr        <- 0.5
pred_train <- ifelse(model$net.result[[1]][,1] > thr, "no", "yes")
tb_train   <- table(y_train, pred_train)
tb_train
1-sum(diag(tb_train))/sum(tb_train)

out_test  <- compute(model, df_test)$net.result
pred_test <- ifelse(out_test[,1] > thr, "no", "yes")
tb_test   <- table(y_test, pred_test)
tb_test
1-sum(diag(tb_test))/sum(tb_test)

#### shallow network 

hidden    <- hidden <- 5
threshold <- 1.2
stepmax   <- 1500 
rep       <- 1
model.shallow     <- neuralnet(model_list, data = df_train, hidden = hidden, threshold = threshold,
                               stepmax = stepmax, rep = rep, lifesign = "minimal", 
                               act.fct = "logistic", linear.output = FALSE)

thr        <- 0.5
pred_train <- ifelse(model.shallow$net.result[[1]][,1] > thr, "no", "yes")
tb_train   <- table(y_train, pred_train)
tb_train
1-sum(diag(tb_train))/sum(tb_train)

out_test_shallow  <- compute(model.shallow, df_test)$net.result
pred_test <- ifelse(out_test[,1] > thr, "no", "yes")
tb_test   <- table(y_test, pred_test)
tb_test
1-sum(diag(tb_test))/sum(tb_test)

#### deep network 

hidden    <- hidden <- c(15, 10, 5, 3)
threshold <- 0.001
stepmax   <- 3000 
rep       <- 5
model.deep     <- neuralnet(model_list, data = df_train, hidden = hidden, threshold = threshold,
                            stepmax = stepmax, rep = rep, lifesign = "minimal", 
                            act.fct = "logistic", linear.output = FALSE)

plot(model.deep)
thr        <- 0.5
pred_train <- ifelse(model.deep$net.result[[1]][,1] > thr, "no", "yes")
tb_train   <- table(y_train, pred_train)
tb_train
1-sum(diag(tb_train))/sum(tb_train)

out_test_deep  <- compute(model.shallow, df_test)$net.result
pred_test_deep <- ifelse(out_test[,1] > thr, "no", "yes")
tb_test_deep   <- table(y_test, pred_test_deep)
tb_test_deep
1-sum(diag(tb_test_deep))/sum(tb_test)

# evaluate deep model
thr <- 0.5
pred_train_deep <- ifelse(model.deep$net.result[[1]][, 1] > thr, "no", "yes")
tb_train_deep <- table(y_train, pred_train_deep)
accuracy_train_deep <- 1 - sum(diag(tb_train_deep)) / sum(tb_train_deep)
cat("Training Accuracy (Deep Model):", accuracy_train_deep, "\n")

out_test_deep <- compute(model.deep, df_test)$net.result
pred_test_deep <- ifelse(out_test_deep[, 1] > thr, "no", "yes")
tb_test_deep <- table(y_test, pred_test_deep)
accuracy_test_deep <- 1 - sum(diag(tb_test_deep)) / sum(tb_test_deep)
cat("Testing Accuracy (Deep Model):", accuracy_test_deep, "\n")
