#install.packages("caret")
library(caret)
#install.packages("dplyr") 
library(dplyr)
#install.packages("psych")
library(psych)
#install.packages("DataExplorer")
library(DataExplorer)
library(ggplot2)
#install.packages("tidyr")
library(tidyr)

#read csv file
data=read.csv('C:/Users/ASUS/Desktop/AML assignment/city_day.csv') 
head(data)
View(data)

#Filter out required data
ban_data = data %>% filter(City == 'Bengaluru')
head(ban_data)
View(ban_data)

#data understanding
dim(ban_data) #dimensions
str(ban_data) #structure

#descriptive statistical analysis
summary(ban_data) #summary
describe(ban_data)

#missing values in a data set
is.na(ban_data) #inconvenient 
sum(is.na(ban_data)) #gives number of missing values
colSums(sapply(ban_data,is.na)) #gives column wise missing values

plot_missing(ban_data) #percentage of missing values in column
remove <- "Xylene"
ban_data = ban_data[, ! names(data) %in% remove, drop = F]
View(ban_data)

#Data exploration using graph/chart
plot_histogram(ban_data) #only numerical data
plot_density(ban_data)
plot_scatterplot(ban_data, 'AQI') #uses two variables.
#
barplot(table(ban_data$AQI), main="AQI", col=c("red")) #plots categorical data
table(ban_data$AQI_Bucket) #number of values per species 

#Handling missing values
str(ban_data)
plot_missing(ban_data)
colSums(sapply(ban_data, is.na))

#Distinct observations present
str(ban_data)
distinct(ban_data)
str(ban_data)

#Drop rows in columns with <5% null values
ban_data=ban_data %>% drop_na(NOx)
sum(is.na(ban_data$NOx))
plot_missing(ban_data)

ban_data=ban_data %>% drop_na(SO2)
sum(is.na(ban_data$S02))
plot_missing(ban_data)

ban_data=ban_data %>% drop_na(CO)
sum(is.na(ban_data$CO))
plot_missing(ban_data)

ban_data=ban_data %>% drop_na(Toluene)
sum(is.na(ban_data$Toluene))
plot_missing(ban_data)

#drop target missing values
ban_data=ban_data %>% drop_na(AQI)
sum(is.na(ban_data$AQI))
plot_missing(ban_data)

#Check skewness of data to decide if mean or median imputation in required
#install.packages('moments')
library(moments) #for skewness function

print(skewness(ban_data$PM2.5)) 
#Skewness function is not working. Therefore, we use describe function given by psych library
describe(ban_data$PM2.5)
hist(ban_data$PM2.5)

describe(ban_data$O3) 
hist(ban_data$O3)

describe(ban_data$Benzene) 
hist(ban_data$Benzene)

describe(ban_data$NH3) 
hist(ban_data$NH3)

describe(ban_data$PM10) 
hist(ban_data$PM10)

#Remaining columns with missing values all have positive skewness.
#When the data is skewed, it is good to consider using the median value for replacing the missing values.
#Mean imputation does not preserve relationship among variables.
#Mean imputation leads to understatement of standard errors.

ban_data$PM2.5[is.na(ban_data$PM2.5)] <- median(ban_data$PM2.5, na.rm = T)
plot_missing(ban_data)

ban_data$PM10[is.na(ban_data$PM10)] <- median(ban_data$PM10, na.rm = T)
ban_data$O3[is.na(ban_data$O3)] <- median(ban_data$O3, na.rm = T)
ban_data$Benzene[is.na(ban_data$Benzene)] <- median(ban_data$Benzene, na.rm = T)
ban_data$NH3[is.na(ban_data$NH3)] <- median(ban_data$NH3, na.rm = T)
plot_missing(ban_data)

#Standardizing the data
#Removing data that is not required for ML model of predicting AQI

ban_data = subset(ban_data, select = -c(City,Date,AQI_Bucket) )
View(ban_data)

skewness(ban_data)

#Min-max scaling
normalize <- function(x, na.rm=TRUE){
  if(is.vector(x)==TRUE){
    maxs <- max(x, na.rm = na.rm)
    mins <- min(x, na.rm = na.rm)
    scale(x,center=mins,scale=maxs-mins)
  } else {
    maxs <- apply(x, 2,max)
    mins <- apply(x, 2,min)
    scale(x, center = mins, scale = maxs - mins)
  }
}

dim(ban_data)
bs=ban_data[,-12] #excluding AQI
str(bs)
bs.norm=as.data.frame(lapply(bs, normalize))
View(bs)
AQI=ban_data$AQI
bs_new=cbind(bs.norm, AQI)


#minmax_data=normalize(ban_data)
#summary(minmax_data)
skewness(bs_new)
summary(bs_new)
#View(minmax_data)

dim(bs_new)
bs1=bs_new[,-12]
str(bs1)
#Log transformation 
bd_final=log(bs1+1)
bandata_final=cbind(bd_final, AQI)

View(bandata_final)
colSums(is.na(bandata_final))
skewness(bandata_final)

#Histogram of columns
par(mfrow=c(2,1))
histout=apply(bandata_final,2,hist)
skewness(bandata_final) #to check if data normalized

#checking linearity of data
library(ggplot2)
ggplot(bandata_final, aes(PM2.5, AQI))+geom_point()+stat_smooth(methold=lm, formula = y ~ x)


#Data split
library(caTools)
set.seed(1)
split=sample.split(bandata_final$AQI, SplitRatio=0.8)
train=subset(bandata_final, split == T)
test=subset(bandata_final, split ==F)
dim(train)
dim(test)
summary(train)

#Model Building
mlr = lm(AQI~. , train)
summary(mlr)

#RMSE and MAE with predict
predict = predict(mlr, test)
predict
library(Metrics)
library(caret)

rmse(test$AQI, predict)
mae(test$AQI, predict)

str(ban_data)
summary(bandata_final)

#rebuild model with significant variables
mlr_1=lm(AQI~ PM2.5+PM10+CO+O3+NO+Toluene, train)
summary(mlr_1)

predict_1 = predict(mlr_1, test)
rmse(test$AQI, predict_1)
mae(test$AQI, predict_1)

#check for overfit
predict_2=predict(mlr_1,train)
str(predict_2)
rmse(train$AQI, predict_2)
mae(train$AQI, predict_2)

mlr_1$coefficients

#Scatter plots to check model 
plot(bandata_final$PM2.5, bandata_final$AQI, main = "Scatterplot")
abline(mlr_1, col=2, lwd=3)
plot(bandata_final$PM10, bandata_final$AQI, main = "Scatterplot")
abline(mlr_1, col=2, lwd=3)
plot(bandata_final$CO, bandata_final$AQI, main = "Scatterplot")
abline(mlr_1, col=2, lwd=3)
plot(bandata_final$O3, bandata_final$AQI, main = "Scatterplot")
abline(mlr_1, col=2, lwd=3)
plot(bandata_final$Toluene, bandata_final$AQI, main = "Scatterplot")
abline(mlr_1, col=2, lwd=3)
plot(bandata_final$NO, bandata_final$AQI, main = "Scatterplot")
abline(mlr_1, col=2, lwd=3)

#--------------------------------------------------------------------------

#SVM

dim(train)
dim(test)

library(e1071)
#default kernel = rbf
svm_rbf=svm(AQI~ PM2.5+PM10+CO+O3+Toluene+NO, train, kernel='radial' )
summary(svm_rbf)
y_pred_rbf=predict(svm_rbf, test)
rmse(test$AQI, y_pred_rbf)
mae(test$AQI, y_pred_rbf)

#default kernel = linear
svm_li=svm(AQI~ PM2.5+PM10+CO+O3+Toluene+NO, train, kernel='linear' )
summary(svm_li)
y_pred_li=predict(svm_li, test)
rmse(test$AQI, y_pred_li)
mae(test$AQI, y_pred_li)

#default kernel = poly
svm_poly=svm(AQI~ PM2.5+PM10+CO+O3+Toluene+NO, train, kernel='poly' )
summary(svm_poly)
y_pred_poly=predict(svm_poly, test)
rmse(test$AQI, y_pred_poly)
mae(test$AQI, y_pred_poly)

#default kernel = sigmoid
svm_sig=svm(AQI~ PM2.5+PM10+CO+O3+Toluene+NO, train, kernel='sigmoid' )
summary(svm_sig)
y_pred_sig=predict(svm_sig, test)
rmse(test$AQI, y_pred_sig)
mae(test$AQI, y_pred_sig)

rmse(test$AQI, y_pred_rbf)
mae(test$AQI, y_pred_rbf)
rmse(test$AQI, y_pred_li)
mae(test$AQI, y_pred_li)
rmse(test$AQI, y_pred_poly)
mae(test$AQI, y_pred_poly)
rmse(test$AQI, y_pred_sig)
mae(test$AQI, y_pred_sig)

#Radial is best model from SVM 

#CV -- K-Fold -- K=10
custom=trainControl(method="repeatedcv", repeats=3,
                    number=10,
                    verboseIter=T)

svm_cv = train(AQI~ PM2.5+PM10+CO+O3+Toluene+NO, data=train, method = "svmRadial", trControl=custom)
svm_cv
y_pred_cv=predict(svm_cv, test)
rmse(test$AQI, y_pred_cv)
mae(test$AQI, y_pred_cv)

#overfit check
y_pred_cv_train=predict(svm_cv, train)
rmse(train$AQI, y_pred_cv_train)
mae(train$AQI, y_pred_cv_train)

#APPENDIX 1
final_model=svm_cv$finalModel
final_model
View(test)
dim(test)
#y_pred_fi = predict(final_model, test[-12])
#rmse(test$AQI, y_pred_fi)
#mae(test$AQI, y_pred_fi)

#HP tuning
#HP tuning
svm_tuned=tune(svm, AQI~ PM2.5+PM10+CO+O3+Toluene+NO, data=train, trainControl=custom,
               ranges=list(epsilon=seq(0,1,0.1),cost=2^(0:2)
                           ,kernel=c('radial','linear','poly')))
summary(svm_tuned)

opt.model=svm_tuned$best.model
summary(opt.model)
y_pred_opt=predict(opt.model, test)
rmse(test$AQI, y_pred_opt)
mae(test$AQI, y_pred_opt)

#overfit check
predict_3=predict(opt.model,train)
RMSE(predict_2, train$AQI)
MAE(predict_2, train$AQI)

#----------------------------------------------------------------------------------------------------------

#Regression Tree


library(rpart)

remove.packages('rpart.plot')
remove.packages('rpart')
remove.packages('rattle')
remove.packages('RColorBrewer')

install.packages('rattle')
install.packages('rpart')
install.packages('rpart.plot')
install.packages('RColorBrewer')

library(rattle)
library(RColorBrewer)
library(rpart.plot)
library(party)

"Can generate different types of trees with rpart
Default split is with Gini index"

tree = rpart(AQI~ PM2.5+PM10+CO+O3+Toluene+NO, data=train)
tree
prp(tree) # plot Rpart Model
prp (tree, type = 5, extra = 100)
rpart.plot(tree, extra = 100, nn = TRUE)
printcp(tree)

# Split with entropy information
ent_Tree = rpart(AQI~ PM2.5+PM10+CO+O3+Toluene+NO, data=train, method="anova", parms=list(split="information"))
ent_Tree
prp(ent_Tree)


library(rpart.plot)
plotcp(tree)

# Here we use tree with parameter settings.
# This code generates the tree with training data
tree_with_params = rpart(AQI~ PM2.5+PM10+CO+O3+Toluene+NO, data=train, method="anova", minsplit = 1, minbucket = 10, cp = 0)
prp (tree_with_params)
print(tree_with_params)
summary(tree_with_params)
plot(tree_with_params)
text(tree_with_params)
plotcp(tree_with_params)

# Now we predict and evaluate the performance of the trained tree model 
predict_4 = predict(tree, test)
# Now examine the values of Predict. These are the class probabilities
predict_4
table(predict_4, test$AQI) 

# Evaluation Measure 
library(caret)
summary(predict_4)
RMSE(predict_4, test$AQI)
# MSE = RMSE^2
MAE(predict_4, test$AQI)
varImp(tree)

#overfit check
predict_5=predict(tree,train)
RMSE(predict_5, train$AQI)
MAE(predict_5, train$AQI)

#ent_Tree - tree  with entropy information
predict_6 = predict(ent_Tree, test)
RMSE(predict_6, test$AQI)
MAE(predict_6, test$AQI)
varImp(ent_Tree)

#overfit check
predict_7=predict(ent_Tree,train)
RMSE(predict_7, train$AQI)
MAE(predict_7, train$AQI)

#tree_with_params - tree with parameter settings
predict_8 = predict(tree_with_params, test)
RMSE(predict_8, test$AQI)
MAE(predict_8, test$AQI)
varImp(tree_with_params)

#overfit check
predict_9=predict(tree_with_params,train)
RMSE(predict_9, train$AQI)
MAE(predict_9, train$AQI)

#test error is comparatively very high to train and therefore, model could be overfit
#-------------------------------------------------------------------------------------------------------

#ANN
library(neuralnet)
nn_1 = neuralnet(AQI~ PM2.5+PM10+CO+O3+Toluene+NO, train, hidden = 10, err.fct='sse', linear.output = F) #err.fct = 'ce' cross entropy for classification//regression regression err.fct = 'sse'
plot(nn_1)
summary(nn_1)
nn_1$result.matrix

predict_10 = predict(nn_1, test)
predict_10
RMSE(predict_10, test$AQI)
MAE(predict_10, test$AQI)

#HYPER PARAMETERS
#decay (drop out) HP is regularization for overfit
#size is number of units in the hidden layer and nnet library supports only 1 hidden layer

library(nnet)
library(caret)

nn_tune = train(AQI~ PM2.5+PM10+CO+O3+Toluene+NO, train, method = 'nnet', metric = 'RMSE',
                tunegrid=expand.grid(size=c(1:10),
                                     decay = c(0,0.1,0.5,1,1.5,2)))
nn_tune
plot(nn_tune)
summary(nn_tune)

predict_11 = predict(nn_tune, test)
predict_11
RMSE(predict_11, test$AQI)
MAE(predict_11, test$AQI)

#overfit check
predict_12=predict(nn_1,train)
RMSE(predict_12, train$AQI)
MAE(predict_12, train$AQI)

#-----------------------------------------------------------------------------------------------------------------

library(randomForest)
rf=randomForest(AQI~ PM2.5+PM10+CO+O3+Toluene+NO,train)
print(rf)

rf$ntree #by default = 500 #HP1
rf$mtry #HP2

predict_13=predict(rf, test)
library(caret)
RMSE(predict_13, test$AQI)
MAE(predict_13, test$AQI)

#tuning
tune.grid=expand.grid(mtry=c(1:8))

rf_tune=train(AQI~ PM2.5+PM10+CO+O3+Toluene+NO,train,method='rf',metric='RMSE', tuneGrid = tune.grid)
print(rf_tune)
rf_tune_best=rf_tune$finalModel
predict_14=predict(rf_tune_best, test)
RMSE(predict_14, test$AQI)
MAE(predict_14, test$AQI)


#Gradient Boosting
gbm=train(AQI~ PM2.5+PM10+CO+O3+Toluene+NO,train,method='gbm',metric='RMSE')
#?gbm #CODE FOR CHECKING DOCUMENTATION
gbm
predict_15=predict(gbm, test)
RMSE(predict_15, test$AQI)
MAE(predict_15, test$AQI)


#Xtreme-GB (XGB)
#install.packages("xgboost")
library(xgboost)
xbm=train(AQI~ PM2.5+PM10+CO+O3+Toluene+NO,train,method='xgbTree',metric='RMSE')
xbm
predict_16=predict(xbm, test)
RMSE(predict_16, test$AQI)
MAE(predict_16, test$AQI)

#boosting algorithm does not give best result. tuned algorithm is considered instead

predict_17=predict(rf_tune_best, train)
RMSE(predict_17, train$AQI)
MAE(predict_17, train$AQI)

#tuned RF gives us overfit model

#bagging in performed instead
library(ipred)
library(rpart)
library(MASS)
library(TH.data)
?bagging

gbag = bagging(AQI~ PM2.5+PM10+CO+O3+Toluene+NO, train, coob=TRUE)
print(gbag)
predict_18=predict(gbag, test)
RMSE(predict_18, test$AQI)
MAE(predict_18, test$AQI)

predict_19=predict(gbag, train)
RMSE(predict_19, train$AQI)
MAE(predict_19, train$AQI)

gbag$err
#overfiting not present anymore
#bagged model is considered
#--------------------------------------------------------------------------------------------------------------------------------
