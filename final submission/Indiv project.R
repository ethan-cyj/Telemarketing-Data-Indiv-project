#1. data wrangling
rm(list = ls())
setwd("~/Documents/DSE1101/indiv project")
#install.packages(c("ROCR", "kknn","boot", "corrplot", "rpart", "MASS", "rpart.plot"))

library(corrplot) #for correlation visualization
library(ROCR) #for roc curves
library(kknn) #for knn
library(boot) #for cv 
library(rpart)#for CART
library(MASS) #stepAIC for var selection
library(e1071) #for NB


bank = read.csv("~/Documents/DSE1101/indiv project/bank-additional.csv", sep=";")
summary(bank)
sum(is.na(bank)) #checking for na values other than unknowns
bank$y1 = ifelse(bank$y == "yes", 1 ,0) #turn dependent var into binary
bank = subset(bank, select = -c(y))
sum(duplicated(bank)[1:22]) #check for duplicated 
unknowns = which(bank == "unknown", arr.ind = TRUE) 
if_minus_unknowns = bank[-unknowns,] #how will our sample size be reduced if we remove unknowns from categorical predictors
new_bank = bank #after these transofrmations we call the dataset "new_bank"




#2. make sense of variables)
attach(new_bank)

#education
loner = which(new_bank == "illiterate", arr.ind = TRUE)#identified only one observationj of "illiterate" class
loner #for simplification purposes, we will no longer record "illiterate", however we can represent it with 0 for future entries
new_bank = new_bank[-3927,]  #only one observation, good chance that it appears in testing set and not training set later
new_bank$education = factor(new_bank$education, c("basic.4y",
                                            "basic.6y","basic.9y","high.school",
                                     "professional.course","university.degree")) #reordering factor levels
new_bank$education = as.numeric(as.factor(new_bank$education))#education turns education level predictor into more ordinal, numeric form
new_bank$education= (new_bank$education-1/2)/7 #rescaling applied

edu_na =which(is.na(new_bank[,4]))# unknown values become NA 
mean_missing_edu =mean(new_bank$education, na.rm = TRUE) #mean of all observations in education is 0.5269733
new_bank$education  = ifelse(is.na(new_bank$education),
                                 mean_missing_edu, new_bank$education)# replace unknowns with the mean
fit_edu = glm(y1~ education, data = new_bank, family = binomial)
summary(fit_edu)

#default
fit_default = glm(y1~ default, data = new_bank, family = binomial)
summary(fit_default)
sum(default=="yes")
#since number of defaults == yes is 1, we remove defaults as it may appear in testing split and not training split
new_bank = new_bank[,-5]#remove default

#housing
new_bank$housing=ifelse(new_bank$housing == "yes", 1 ,ifelse(new_bank$housing == "no", 0, NA))
mean_missing_housing =mean(new_bank$housing, na.rm = TRUE) #mean of all observations in housing is 0.5419885
mean_missing_housing
new_bank$housing  = ifelse(is.na(new_bank$housing),
                           mean_missing_housing, new_bank$housing)# replace unknowns with the mean

#loan
new_bank$loan=ifelse(new_bank$loan == "yes", 1 ,ifelse(new_bank$loan == "no", 0, NA))
mean_missing_loan =mean(new_bank$housing, na.rm = TRUE) #mean is 0.5269733
new_bank$loan  = ifelse(is.na(new_bank$loan),
                        mean_missing_loan, new_bank$loan)# replace unknowns with the mean
fit_loan = glm(y1~ loan, data = new_bank, family = binomial)
summary(fit_loan)
#duration
glm_duration = glm(y1~ duration, data = new_bank, family = "binomial")
summary(glm_duration) #coef of 0.0036541, p-val <2e-16 

#do correlation matrix
nums = unlist(lapply(new_bank, is.numeric))  #get numeric
bank_num = new_bank[,nums]#bank_num is dataset with only numeric variables

corr = round(cor(bank_num),2)
corr #correlation between numeric variables
corrplot::corrplot(corr,type= "upper" , order = "hclust", #visualisation tool for correlation
         tl.col = "black", tl.srt = 45 )
title("Figure 3: correlation plot of numeric variables",)
#pdays, previous and poutcome very similar, as pdays ==999, previous ==0 and poutcome==nonexistent all coincide
#previous has high degree of multicollinearity with 4 different predictors, thus we remove

#from cor plot on numeric variables, some variables have v high correlation, there
#is some degree of multicollinearity, euribor3m has correlations of .97 and .94
#with emp.var.rate and nr.employed respectively. Thus by removing the variable it removes the 
#highest amount of multicollinearity 


#glm_final is the linear regression model with the variables we choose to exclude
glm_final = glm(y1~ .- duration -euribor3m-previous, data = new_bank, family = binomial)
summary(glm_final)

#2.5 unsupervised methods

#PCA
pcntrain=round(nrow(bank_num)*0.5,0) #use 50/50 split
set.seed(100)
pctr = sample(1:nrow(bank_num),pcntrain)  # draw ntrain observations from original data
pctraining = bank_num[pctr,]   # Training split
pctesting = bank_num[-pctr,] #testing split
prall = prcomp(bank_num, scale = TRUE) #performing pca
#Now produce the biplot using the biplot() function on the prcomp object:
biplot(prall, main="Figure 5: biplot")
prall.s = summary(prall)
prall.s$importance
scree = prall.s$importance[2,] #save the proportion of variance explained
plot(scree, main = "Figure 4: Scree Plot of Principal Components", xlab = "Principal Component", 
     ylab = "Proportion of Variance Explained",ylim =c(0,1), type = 'b', cex = .8)
#plot the scree plot

library(pls)
pcr.fit=pcr(y1~.-duration,data=pctraining, scale=TRUE, validation="CV")#perform pcr
summary(pcr.fit)
validationplot(pcr.fit, val.type="MSEP", main="LOOCV",legendpos = "topright") #lowest ncomp =11

pcr.pred=predict(pcr.fit, newdata=pctesting, ncomp=11, type='response')#validation set testing of pcr.fit
mean((pctesting$y1-pcr.pred)^2) #MSE for PCR 0.07749517
pcr_confusion = table(pcr.pred > 0.5, pctesting$y1)
pcr_confusion
pcr_accuracy = sum(diag(pcr_confusion)) / sum(pcr_confusion)
pcr_accuracy#  0.9062652   using standardised cutoff 0.5
pcr_pred = prediction(as.numeric(pcr.pred), pctesting$y1)
pcr_perf = performance(pcr_pred, measure = "tpr", x.measure = "fpr")
plot(pcr_perf)
pcr_auc = performance(pcr_pred, measure = "auc")
pcr_auc@y.values #AUC of 0.7320849



#3. modelling- find best predictors model selection
#stepwise model selection using AIC
stepAIC(glm_final, direction = c("backward"), trace = TRUE)

# AIC can measure model fit, with a penalizing factor for number of predictors. smaller better 
#according to the stepwise selection, the following predictors minimised the AIC value
glm_selected = glm(y1 ~ age+ contact + month + campaign + poutcome + emp.var.rate + 
      cons.price.idx + cons.conf.idx, family = "binomial", data = new_bank)

#perform split of entire dataset
ntrain=round(nrow(new_bank)*0.5,0) #use 50/50 split
set.seed(100)
tr = sample(1:nrow(new_bank),ntrain)  # draw ntrain observations from original data
training = new_bank[tr,]   # Training split
testing = new_bank[-tr,] #testing split

#4. evaluation : roc, auc, visualisation, cost function?
#training "hand selected" regression
glm1_fit = glm(y1 ~.-duration -euribor3m -previous, data = training, family = "binomial")
glm1_prob = predict(glm1_fit, newdata = testing, type = "response")
mean((testing$y1-glm1_prob)^2) #MSE of 0.0742924 
glm1_confusion = table(glm1_prob > 0.5, testing$y1)
glm1_confusion
glm1_accuracy = sum(diag(glm1_confusion)) / sum(glm1_confusion)
glm1_accuracy#  0.9067508  using standardised cutoff 0.5
glm1_pred = prediction(glm1_prob, testing$y1)
glm1_perf = performance(glm1_pred, measure = "tpr", x.measure = "fpr")
plot(glm1_perf)#roc curve
glm1_auc = performance(glm1_pred, measure = "auc")
glm1_auc@y.values # AUC is 0.7575291
#if predictor "duration" is included, acc=  0.90162 (5sf) using standardised cutoff 0.5, auc = 0.9196125
#"duration" is not a realistic predictor to be included. Even though it is a strong predictor of y, 
# it is only measured after the call has been made, after which y is known, so banks cannot construct a model which includes it.
#excluding "duration", we have accuracy = 0.88432 (5sf), and AUC = 0.73242 (5sf)

# training "optimal" logistic regression model
glm_fit = glm(y1 ~ age+ contact + month + campaign + poutcome + emp.var.rate + 
                cons.price.idx + cons.conf.idx-1, data = training, family = "binomial")
summary(glm_fit)
#Predict the test observations
glm_prob = predict(glm_fit, newdata = testing, type = "response")
mean((testing$y1-glm_prob)^2) #MSE of 0.07365964 
#Build the confusion matrix:
glm_confusion = table(glm_prob > 0.5, testing$y1)
glm_confusion
glm_accuracy = sum(diag(glm_confusion)) / sum(glm_confusion) #0.9082079 using standardised cutoff 0.5
glm_accuracy
glm_pred = prediction(glm_prob, testing$y1)
glm_perf = performance(glm_pred, measure = "tpr", x.measure = "fpr")
glm_auc = performance(glm_pred, measure = "auc")
glm_auc@y.values 
#auc of 0.7771693, marked improvement over glm_1

#finding max accuracy at optimal threshold
glm_accuracy_perf = performance(glm_pred, measure = "acc")
plot(glm_accuracy_perf, col = "deeppink3", lwd = 2)
glm_ind = which.max(slot(glm_accuracy_perf, "y.values")[[1]])
glm_cutoff = slot(glm_accuracy_perf, "x.values")[[1]][glm_ind]
glm_acc = slot(glm_accuracy_perf, "y.values")[[1]][glm_ind]
glm_acc
#at optimal cutoff, slight improvement in accuracy to 0.9096649 (5sf)
#solid model in my opinion. must consider high correlation between "emp.var.rate" 
#and "nr.employed","cons.price.idx", which might inflate predictive power of the model



#4. Naïve Bayes Classifier
#NB with selected variables
nbtrain1 = naiveBayes(y1 ~ .-duration-euribor3m-previous, data =training)
nbprob1 = predict(nbtrain1, testing, type = "class")
nbprobraw1 = predict(nbtrain1, testing, type = "raw")
mean((testing$y1-nbprobraw1[,2])^2) #MSE of 0.1074117
nb_confusion1= table(nbprob1, testing$y1)
nb_confusion1
nb_accuracy1 = sum(diag(nb_confusion1)) / sum(nb_confusion1) #accuracy of 0.8776105 
nb_accuracy1
nbpred1 = prediction(nbprobraw1[,2], testing$y1)
nbperf1 = performance(nbpred1, measure = "tpr", x.measure = "fpr")
nbauc1 = performance(nbpred1, measure ="auc")
nbauc1@y.values #NB gives auc of 0.7582338


#5. knn
# TO DO: remove binary variables before we can use knn
new_bank.loocv=train.kknn(y1 ~ age+  education + campaign + pdays + emp.var.rate + 
                            cons.price.idx + cons.conf.idx+ nr.employed, 
                          data =training, kmax=100, kernel = "rectangular")
#numeric+ scaled ordinal variables minus previoux, minus euribor3m to reduce effect of multicollinearity

kbest=new_bank.loocv$best.parameters$k
knnpredcv=kknn(y1 ~ age+ education + campaign + previous + emp.var.rate + 
                 cons.price.idx + cons.conf.idx+ nr.employed,
               training,testing,k=kbest,kernel = "rectangular")
mean((testing$y1-knnpredcv$fitted.values)^2) #MSE of 0.07880525
table(knnpredcv$fitted.values>0.5,testing$y1) #confusion matrix?
knnpred = prediction(knnpredcv$fitted.values, testing$y1)
knnperf = performance(knnpred, measure = "tpr", x.measure = "fpr")
knn_auc = performance(knnpred, measure = "auc")
knn_auc@y.values# 0.7373676 


#6. decision tree
treeGini= rpart(y1~.-duration,
                data=training, method = "class", minsplit = 10, cp = .000001, maxdepth = 30)
plotcp(treeGini)
bestcp=treeGini$cptable[which.min(treeGini$cptable[,"xerror"]),"CP"]#find optimal cp which is about 0.028571
bestgini = prune(treeGini, cp = bestcp)#this is the optimal tree
treeprob = predict(bestgini, newdata = testing)
plot(bestgini, uniform = TRUE, main ="Regression Tree") 
text(bestgini,digits=4,use.n=TRUE,fancy=FALSE,bg='lightblue')
#can see tht the tree chose nr.employed and pdays, good to know as the variables selected by tree can help us with vairbale selectopn
mean((testing$y1-treeprob[,2])^2) #MSE of 0.07738495 
treepred = prediction(treeprob[,2], testing$y1)
treeperf = performance(treepred, measure = "tpr", x.measure = "fpr")
treeauc = performance(treepred, measure = "auc")
treeauc@y.values #auc of 0.6920829 
#visualising tree
library(rpart.plot)
rpart.plot(bestgini, shadow.col = "gray",type =5, extra =2, )
title("Figure 6: Regression Tree")


#comparing performance
plot(glm1_perf, col = "blue", lwd = 1, main= "Figure 2: Comparison of ROC curves for diferent models") #glm using ethan's big brain intuition 0.758
plot(glm_perf, col = "purple", lwd = 1,add = TRUE ) #glm using aic to perform stepwise selection 0.777
plot(nbperf1, col = "red", lwd = 1,add = TRUE )#using Naïve Bayes Classifier 0.758
plot(knnperf, col = "orange", lwd = 1, add= TRUE) #using K nearest neighbours 0.737
plot(treeperf, col = "green", add= TRUE) #decision tree 0.692
plot(pcr_perf,col = "black", add=TRUE) #PCR. auc of 0.732
abline(0, 1, lwd = 1, lty = 2)
legend("bottomright", legend=c("LR_1: 0.758", "LR_2: 0.777","NBC:  0.758", "KNN:  0.737","DT:     0.692", "PCR:  0.732"),
       col=c("blue", "purple", "red", "orange", "green", "black"),title ="AUC values", lty=1, cex=0.8)#legend

#other writing points:
#costs of fpr &fnr
#suspect that companies would prefer a sensitive model, as the cost of failure after calling probably not as high opp cost of not calling
#variable selection based on LR performance then projected onto other methods, possibly not the best but used to standardise for comparison

