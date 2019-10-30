#==================Coursework 3===========================
#==================Submitted By===========================
#==================Rubaida Easmin=========================

#==================remove workspace=======================
rm(list=ls())

#===================directory=============================
#setwd("/home/reasm001/RCode")
setwd("C:/Users/eruba/Documents/ML3/")


#==================read trainset==========================
train_x=read.csv("train.csv") 
dim(train_x)
head(train_x)
colnames(train_x)

#=================missing value===========================
colSums(is.na(train_x))
summary(train_x)
train_x$target = as.factor(train_x$target)
train_x = train_x[,c(which(colnames(train_x)=="target"),which(colnames(train_x)!="target"))]
sum(sapply(train_x, is.factor))

#=================near zero variance======================
library(caret)
predictorInfo = nearZeroVar(train_x,saveMetrics = TRUE)
rownames(predictorInfo)[predictorInfo$nzv]         

#=================split trsin/test set====================
set.seed(234)
train_x = train_x[sample(nrow(train_x)),]                                 # Randomly shuffle the data
split = createDataPartition(y=train_x$target, p = 0.80,list = FALSE)
df.train = train_x[split,]
df.test = train_x[-split,]

dim(df.train)
dim(df.test)

#======================correlation=======================
library(corrplot)
df.train_num = df.train[sapply(df.train, is.numeric)]         # find numeric dataset
correlation_dat = cor(df.train_num) #calculate correlation on numeric values
corrplot.mixed(correlation_dat)                                # plot correlation matrix
corr_colnames = findCorrelation(correlation_dat,cutoff = 0.75,names = TRUE) #column names above threshold (8 columns)

#======================skewness===========================
library(tidyr)
library(ggplot2)
ggplot(gather(df.train[,-1][,2:21]), aes(value)) + 
  geom_histogram(bins = 10) + 
  facet_wrap(~key, scales = 'free_x')


#=====================Relief============================
library(AppliedPredictiveModeling)
library(MASS)
library(CORElearn)
reliefVal = attrEval("target",data = df.train,
                     estimator = "ReliefFequalK",
                     ReliefIterations = 50)
reliefVal.col = names(sort(abs(reliefVal), decreasing = TRUE)[1:200])
reliefVal.col

reliefPerm = permuteRelief(x = df.train[,-1],y = df.train$target,
                           nperm = 100, estimator = "ReliefFequalK",
                           ReliefIterations = 50)
perm.col=names(sort(abs(reliefPerm$standardized[which(abs(reliefPerm$standardized)>=1.6)]), decreasing=T))
perm.col

#====================keep column(Relief/ Perm Relief)========================
##without balance dataset top 100 columns
keep.col = c("target","var_22","var_9","var_111","var_45","var_164","var_40","var_20","var_50","var_55",
             "var_177","var_109","var_33","var_13","var_98","var_5","var_197","var_187","var_115",
             "var_64","var_30","var_126","var_125","var_81","var_168","var_101","var_124","var_11",
             "var_26","var_73","var_195","var_63","var_69","var_136","var_181","var_184","var_77",
             "var_10","var_28","var_37","var_149","var_29","var_70","var_85","var_108","var_58",
             "var_15","var_27","var_52","var_173","var_25","var_24","var_7","var_79","var_21",
             "var_38","var_138","var_97","var_86","var_123","var_72","var_91","var_131","var_146",
             "var_140","var_135","var_172","var_121","var_163","var_112","var_78","var_165","var_12",
             "var_80","var_161","var_193","var_190","var_39","var_169","var_82","var_74","var_90",
             "var_186","var_0","var_133","var_87","var_150","var_179","var_57","var_189","var_156",
             "var_92","var_175","var_167","var_53","var_42","var_16","var_159","var_103","var_139","var_67")

df.train = df.train[keep.col]
df.test = df.test[keep.col]

#===============Check balance of dataset==================
barplot(prop.table(table(df.train$target)),xlab = "Distribution of target variable",ylab = "Frequency of the sample")
#install.packages("ROSE")
library(ROSE)

df.train_up = ovun.sample(target ~ ., data = df.train, method = "over",N = 287844)$data
table(df.train_up$target)
barplot(prop.table(table(df.train_up$target)),xlab = "Distribution of target variable",ylab = "Frequency of the sample")


#===============Use VIF for multicolinearity==============
set.seed(123)
logit = glm(target~.,data = df.train_up, family = binomial) # CREATE LOGISTIC MODEL WITH ALL VARIABLES
vif.value = DAAG::vif(logit)


#================Feature Selection (LASSO Regression)=====
library(glmnet)
set.seed(123)
x = model.matrix(target~., df.train_up)[,-1]
y = ifelse(df.train_up$target == "1", 1, 0)
cv.lasso = cv.glmnet(x, y, alpha = 1, family = "binomial")
plot(cv.lasso)
print(cv.lasso$lambda.min)
print(coef(cv.lasso, cv.lasso$lambda.min))
lasso.col = coef(cv.lasso, cv.lasso$lambda.min)

remove_col = c("var_30")
df.train_up = df.train_up[, !(colnames(df.train_up) %in% remove_col)]
dim(df.train_up)

#=======================p-value=====================================
set.seed(123)
logit = glm(target~.,data = df.train_up, family = binomial) # CREATE LOGISTIC MODEL WITH ALL VARIABLES
p.imp = data.frame(summary(logit)$coef[summary(logit)$coef[,4] <= .05, 4])
p.imp
dim(p.imp)

#remove 4 columns
#95 columns
keep.pvalue.col = c("target","var_22","var_9","var_111","var_45","var_164","var_40","var_20","var_50","var_55","var_177","var_109","var_33","var_13",
                    "var_98","var_5","var_197","var_187","var_115","var_64","var_125","var_81","var_168","var_101","var_124","var_11","var_26",
                    "var_73","var_195","var_63","var_69","var_136","var_181","var_184","var_77","var_10","var_28","var_37","var_149","var_29",
                    "var_70","var_85","var_108","var_58","var_15","var_27","var_52","var_173","var_25","var_24","var_79","var_21","var_38",
                    "var_138","var_97","var_86","var_123","var_72","var_91","var_131","var_146","var_140","var_135","var_172","var_121","var_163",
                    "var_112","var_78","var_165","var_12","var_80","var_161","var_193","var_190","var_39","var_169","var_82","var_74","var_90",
                    "var_186","var_0","var_133","var_87","var_150","var_179","var_57","var_189","var_156","var_92","var_175","var_167","var_53",
                    "var_42","var_159","var_139","var_67")

df.train = df.train_up[keep.pvalue.col]
df.test = df.test[keep.pvalue.col]


#============================RFE===============================
control = rfeControl(functions=ldaFuncs, method="cv", number=5)
results = rfe(df.train[,2:96], df.train[,1], sizes=c(2:96), rfeControl=control)
print(results)
predictors(results)

#56 columns
keep.rfe = c("target","var_81","var_139","var_12","var_146","var_21","var_53","var_22","var_26","var_190","var_13","var_165",
             "var_80","var_133","var_115","var_149","var_169","var_0","var_78","var_184","var_40","var_67","var_109",
             "var_92","var_179","var_33","var_108","var_9","var_172","var_173","var_87","var_121","var_123","var_164",
             "var_91","var_86","var_177","var_186","var_131","var_197","var_163","var_167","var_90","var_5","var_52",
             "var_125","var_195","var_70","var_135","var_58","var_24","var_28","var_150","var_112","var_111","var_45")

df.train = df.train[keep.rfe]
df.test = df.test[keep.rfe]

dim(df.train)
dim(df.test)

#==================Feature Selection (Random Forest)===========
library(randomForest)
dfc.train.rf = randomForest(target ~ ., data=df.train, mtry=8,
                            importance=TRUE, na.action=na.omit)
varImpPlot(dfc.train.rf)
rf.feature = dfc.train.rf$importance


#===================Model Analysis========================
#===================Logistic Regression===================
library(pROC)
logit = glm(target~.,data = df.train, family = binomial)
glm.probs = predict (logit ,df.test, type="response")
glm.pred <- ifelse(glm.probs > 0.5, "1", "0")
mean(glm.pred==df.test$target)
confusionMatrix(factor(glm.pred),factor(df.test$target))
#AUCScore
auc(as.numeric(glm.pred), as.numeric(df.test$target))

#=====================Decision Tree=======================
ctrl = trainControl(method = "cv", number = 5)
ctrl2 = trainControl(method = "cv", number = 15)
ctrl3 = trainControl(method = "repeatedcv", number = 10,
                     repeats = 10)
#========CV, k=5
set.seed(231)
dtree_fit = train(target ~., data = df.train, method = "rpart",
                  #parms = list(split = "information"),
                  preProcess=c("center", "scale","BoxCox"),
                  trControl= ctrl,
                  tuneLength = 10)
dtree_fit
test_pred = predict(dtree_fit, newdata = df.test)
confusionMatrix(test_pred, df.test$target)  

#AUC Score
auc(as.numeric(test_pred), as.numeric(df.test$target))


#========CV, k=15
set.seed(231)
dtree_fit = train(target ~., data = df.train, method = "rpart",
                  #parms = list(split = "information"),
                  preProcess=c("center", "scale","BoxCox"),
                  trControl= ctrl,
                  tuneLength = 10)
dtree_fit
test_pred = predict(dtree_fit, newdata = df.test)
confusionMatrix(test_pred, df.test$target)  

#AUC Score
auc(as.numeric(test_pred), as.numeric(df.test$target))

#========Repeated CV, k=10
set.seed(231)
dtree_fit = train(target ~., data = df.train, method = "rpart",
                  #parms = list(split = "information"),
                  preProcess=c("center", "scale","BoxCox"),
                  trControl= ctrl3,
                  tuneLength = 10)
dtree_fit
test_pred = predict(dtree_fit, newdata = df.test)
confusionMatrix(test_pred, df.test$target)  

#AUC Score
auc(as.numeric(test_pred), as.numeric(df.test$target))


#=====================Random Forest=======================
set.seed(231)
ctrl = trainControl(method = "cv", number = 5)
tunegrid=expand.grid(.mtry=7)

#=======CV, K=3
rf_fit = train(target~., data=df.train, 
               method='rf',  
               metric='Accuracy', 
               preProcess=c("center", "scale","BoxCox"),
               tuneGrid=tunegrid, 
               trControl=ctrl)

rf.predict=predict(rf_fit, newdata = df.test)
confusionMatrix(rf.predict, df.test$target)

#AUC Score
auc(as.numeric(rf.predict), as.numeric(df.test$target))



#==================K-Nearest-Neighbour====================
set.seed(231)
library("class")
#========== K=3 
knn_fit = knn(df.train, df.test, df.train$target, k = 3)
confusionMatrix(knn_fit ,df.test$target)

#AUC Score
auc(as.numeric(knn_fit), as.numeric(df.test$target))
roccurve = roc(as.numeric(knn_fit), as.numeric(df.test$target))
plot(roccurve)

#========== K=5
knn_fit2 = knn(df.train, df.test, df.train$target, k = 5)
confusionMatrix(knn_fit2 ,df.test$target)

#AUC Score
auc(as.numeric(knn_fit2), as.numeric(df.test$target))

#========== K=10
knn_fit3 = knn(df.train, df.test, df.train$target, k = 10)
confusionMatrix(knn_fit3 ,df.test$target)

#AUC Score
auc(as.numeric(knn_fit3), as.numeric(df.test$target))


#=====================Naive Based Algorithm=================
library(klaR)
library(pROC)
ctrl = trainControl(method = "cv", number = 5)
ctrl2 = trainControl(method = "cv", number = 15)
ctrl3 = trainControl(method = "repeatedcv", number = 10,
                     repeats = 10)

set.seed(123)
nb_fit = train(df.train[,-1], df.train$target, method="nb", 
               preProcess=c("center", "scale","BoxCox","spatialSign"),
               trControl=ctrl)
nb_pred = predict(nb_fit, df.test)
confusionMatrix(nb_pred, df.test$target, positive="1")

#AUC Score
auc(as.numeric(nb_pred), as.numeric(df.test$target))

#=============CV, K =15
set.seed(123)
nb_fit = train(df.train[,-1], df.train$target, method="nb", 
               preProcess=c("center", "scale","BoxCox","spatialSign"),
               trControl=ctrl2)
nb_pred2 = predict(nb_fit, df.test)
confusionMatrix(nb_pred2, df.test$target, positive="1")

#AUC Score
auc(as.numeric(nb_pred2), as.numeric(df.test$target))

#============Repeated CV, K =10
set.seed(123)
nb_fit = train(df.train[,-1], df.train$target, method="nb", 
               preProcess=c("center", "scale","BoxCox","spatialSign"),
               trControl=ctrl3)
nb_pred3 = predict(nb_fit, df.test)
confusionMatrix(nb_pred3, df.test$target, positive="1")

#AUC Score
auc(as.numeric(nb_pred3), as.numeric(df.test$target))


#===================Extreme Gradient Descent==============
library(xgboost)
library(pROC)
ctrl = trainControl(method = "cv", number = 5)
#=====CV, K=5
X_train = xgb.DMatrix(as.matrix(df.train[,-1]))
y_train = df.train$target
X_test = xgb.DMatrix(as.matrix(df.test[,-1]))
y_test = df.test$target

set.seed(123) 
xgb_model = train(
  X_train, y_train,  
  trControl = ctrl,
  #tuneGrid = xgbGrid,
  method = "xgbTree"
)

predicted = predict(xgb_model, X_test)
confusionMatrix(predicted,y_test)

auc(as.numeric(predicted), as.numeric(y_test))


#=========================Elastic Net=====================
library(glmnet)
library(pROC)
set.seed(231)

#======CV, k=5
cv_5 = trainControl(method = "cv", number = 5)
elnet_int = train(
  target ~ ., data = df.train,
  method = "glmnet",
  trControl = cv_5,
  tuneLength = 10
)
elnet_int$bestTune
coef(elnet_int$finalModel, elnet_int$bestTune$lambda)


elastic_net.pred = predict(elnet_int,df.test)
confusionMatrix(elastic_net.pred,df.test$target)
auc(as.numeric(elastic_net.pred), as.numeric(df.test$target))

#======CV, k=15
set.seed(231)
cv_5 = trainControl(method = "cv", number = 15)
elnet_int = train(
  target ~ ., data = df.train,
  method = "glmnet",
  trControl = cv_5,
  tuneLength = 10
)
elnet_int$bestTune
coef(elnet_int$finalModel, elnet_int$bestTune$lambda)


elastic_net.pred = predict(elnet_int,df.test)
confusionMatrix(elastic_net.pred,df.test$target)
auc(as.numeric(elastic_net.pred), as.numeric(df.test$target))


#======Repeated CV, k=10
set.seed(231)
cv_5 = trainControl(method = "repeatedcv",
                    number = 10,
                    repeats = 10)
elnet_int = train(
  target ~ ., data = df.train,
  method = "glmnet",
  trControl = cv_5,
  tuneLength = 10
)
elnet_int$bestTune
coef(elnet_int$finalModel, elnet_int$bestTune$lambda)


elastic_net.pred = predict(elnet_int,df.test)
confusionMatrix(elastic_net.pred,df.test$target)
auc(as.numeric(elastic_net.pred), as.numeric(df.test$target))
