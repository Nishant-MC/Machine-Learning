library(e1071)

train <- read.table("normalized_mnist_train.txt", header = TRUE, sep = ",")
test <- read.table("normalized_mnist_test.txt", header = TRUE, sep = ",")

train_y <- train[1]
train[1] = NULL
test_y <- test[1]
test[1] = NULL
mysvm = svm(train,train_y,type='C',kernel='linear')
# error rate is 0.108 
mysvm = svm(train,train_y,type='C',kernel='polynomial',degree=5)
# error rate is 0.061
mysvm = svm(train,train_y,type='C',kernel='radial', gamma=0.001, cost=16)
# error rate is 0.082
tuned_svm <- tune.svm(train, train_y, gamma = 2^(-13:-6), cost = 2^(2:4), tunecontrol=tune.control(cross=10))
summary(tuned_svm)
# best parameters are gamma:0.0078125, cost:16, with an error rate of 0.051
pred = predict(mysvm, test)
test_y <- t(test_y)
tbl1 = table(pred, test_y)
error_rate = 1-sum(diag(tbl1))/sum(tbl1)
