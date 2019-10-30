#==================Coursework 3===========================
#==================Submitted By===========================
#==================Rubaida Easmin=========================
#=====Implementation of Principle Component Analysis======




#==================remove workspace=======================
rm(list=ls())

#===================directory=============================
setwd("/home/reasm001/RCode")
#setwd("C:/Users/eruba/Documents/ML3/")

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
train_x = train_x[, !(names(train_x) %in% c("ID_code"))]#remove categorical columns
dim(train_x)

#===================Convert into PCA========================
target = train_x[,1]
data = train_x[,-1]
pca = princomp(data,cor=F)
summary(pca)
plot(pca)
gof = (pca$sdev)^2/sum((pca$sdev)^2)
sum(gof[1:131])

plot(gof, xlab = "Principal Component",
     ylab = "Proportion of Variance Explained",
     type = "b")

plot(cumsum(gof), xlab = "Principal Component",
     ylab = "Cumulative Proportion of Variance Explained",
     type = "b")

newdata = pca$scores[,1:131]
newdata = cbind(target,newdata)
colnames(newdata) = c("target","p1","p2","p3","p4","p5","p6","p7","p8","p9","p10","p11","p12","p13","p14","p15","p16","p17","p18","p19","p20","p21","p22","p23","p24",
                      "p25","p26","p27","p28","p29","p30","p31","p32","p33","p34","p35","p36","p37","p38","p39","p40","p41","p42","p43","p44","p45","p46",
                      "p47","p48","p49","p50","p51","p52","p53","p54","p55","p56","p57","p58","p59","p60","p61","p62","p63","p64","p65","p66","p67","p68",
                      "p69","p70","p71","p72","p73","p74","p75","p76","p77","p78","p79","p80","p81","p82","p83","p84","p85","p86","p87","p88","p89","p90",
                      "p91","p92","p93","p94","p95","p96","p97","p98","p99","p100","p101","p102","p103","p104","p105","p106","p107","p108","p109","p110",
                      "p111","p112","p113","p114","p115","p116","p117","p118","p119","p120","p121","p122","p123","p124","p125","p126","p127","p128","p129","p130","p131")
newdata=as.data.frame(newdata)
head(newdata)
newdata$target = as.factor(newdata$target)

#==============train/test split===========================
library(caret)
set.seed(2345)
partition = createDataPartition(newdata$target,times=1,p=0.7)
train = newdata[partition$Resample1,]
test = newdata[-partition$Resample1,]


#====================Model Analysis=======================
#===================Logistic Regression===================
library(glmnet)
library(pROC)
logit = glm(target~.,data = train, family = binomial)
glm.probs = predict (logit ,test, type="response")
glm.pred <- ifelse(glm.probs > 0.5, "1", "0")
mean(glm.pred==test$target)
confusionMatrix(factor(glm.pred),factor(test$target))
auc_score = auc(as.numeric(glm.pred), as.numeric(test$target))
auc_score


#=====================Random Forest=======================
set.seed(231)
fitControl=trainControl(
  method = "cv",
  number = 10)
tunegrid=expand.grid(.mtry=7)
rf_fit = train(target~., data=train, 
               method='rf',  
               metric='Accuracy', 
               preProcess=c("center", "scale","BoxCox"),
               tuneGrid=tunegrid, 
               trControl=fitControl)

rf.predict=predict(rf_fit, newdata = test)
confusionMatrix(rf.predict, as.factor(test$target))

#AUC Score
auc(as.numeric(rf.predict), as.numeric(test$target))


#=====================Decision Tree=======================
trctrl = trainControl(method = "cv", number = 5)
set.seed(231)
dtree_fit = train(target ~., data = train, method = "rpart",
                  parms = list(split = "information"),
                  preProcess=c("center", "scale","BoxCox"),
                  trControl=trctrl,
                  tuneLength = 10)
dtree_fit
test_pred = predict(dtree_fit, newdata = test)
confusionMatrix(as.factor(test_pred), as.factor(test$target))  

#AUC Score
auc(as.numeric(test_pred), as.numeric(test$target))
roccurve = roc(as.numeric(test_pred), as.numeric(test$target))
plot(roccurve)

#=====================Naive Based Algorithm=================
library(klaR)
library(pROC)
ctrl = trainControl(method="cv", 5)
set.seed(123)
nb_fit = train(train[,-1], train$target, method="nb", 
               preProcess=c("center", "scale","BoxCox","spatialSign"),
               trControl=ctrl)
nb_pred = predict(nb_fit, test)
confusionMatrix(nb_pred, test$target, positive="1")

#AUC Score
auc(as.numeric(nb_pred), as.numeric(test$target))


#==================K-Nearest-Neighbour====================
set.seed(231)
#install.packages("class")
library("class")
library(pROC)
knn_fit = knn(train, test, train$target, k = 3)
confusionMatrix(knn_fit ,test$target)

#AUC Score
auc(as.numeric(knn_fit), as.numeric(test$target))


#===================Extreme Gradient Descent==============
library(xgboost)
library(pROC)
ctrl = trainControl(method = "cv", number = 5)

X_train = xgb.DMatrix(as.matrix(train[,-1]))
y_train = train$target
X_test = xgb.DMatrix(as.matrix(test[,-1]))
y_test = test$target

set.seed(123) 
xgb_model = train(
  X_train, y_train,  
  trControl = ctrl,
  method = "xgbTree"
)

predicted = predict(xgb_model, X_test)
confusionMatrix(predicted,y_test)

auc(as.numeric(predicted), as.numeric(y_test))



#=========================Elastic Net=====================
library(glmnet)
library(pROC)
set.seed(231)
cv_5 = trainControl(method = "cv", number = 10)
elnet_int = train(
  target ~ ., data = train,
  method = "glmnet",
  trControl = cv_5,
  tuneLength = 10
)

elastic_net.pred = predict(elnet_int,test)
confusionMatrix(elastic_net.pred,test$target)

auc(as.numeric(elastic_net.pred), as.numeric(test$target))
