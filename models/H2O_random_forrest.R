
# load the preprocessed data
load("Rossmann.RData")

# when the store is closed, the sales will be 0, so here we will only use sale > 0 
train <- train[Sales > 0,]  

# convert the NA values to 0
train[is.na(train)] <- 0
test[is.na(test)] <- 0

# log transform the sales data
train[,logSales:=log1p(Sales)]

# build random forrest models in H2O package
# load package
library(h2o)
# start cluster with all available threads
h2o.init(nthreads=-1,max_mem_size='6G')
# load train data into cluster from R
train_hex<-as.h2o(train)
# select features for the model
features<-colnames(train)[!(colnames(train) %in% c("Id","Date","Sales","logSales","Customers", "DayOfWeek"))]
# model training
rf_hex <- h2o.randomForest(x=features,
                          y="logSales", 
                          ntrees = 400, ## can be changed to get the best predicting result
                          max_depth = 25, ## can be changed to get the best predicting result
                          nbins_cats = 1115, ## allow it to fit store ID
                          training_frame=train_hex)

# check the model
summary(rf_hex)

# load test data into cluster from R
test_hex<-as.h2o(test)
# predicting in test data
predictions<-as.data.frame(h2o.predict(rf_hex,test_hex))
# return the predictions to the original scale of the Sales data
pred <- expm1(predictions[,1])
# generate submission data
submission <- data.frame(Id=test$Id, Sales=pred)
# save predicted data
write.csv(submission, "h2o_rf_400_25.csv",row.names=F)
# save model
save(rf_hex, file="h2o_rf_400_25.RData")