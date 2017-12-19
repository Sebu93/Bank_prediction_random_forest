########READING CSV###################################################

setwd("C:/Users/sebas/Desktop/Glm_files/Data minig_group")

data_full = read.csv("PL_XSELL.csv")

#######################################################################
library(neuralnet)
library(caret)
library(scales)
library(randomForest)

##############Splitting data############################################

trIndex<-createDataPartition(data_full$TARGET,
                             p = 0.8,
                             list=F)

Dev_sample = data_full[trIndex,]
Hold_sample = data_full[-trIndex,]

write.csv(Dev_sample,file = "Dev_Sample.csv",row.names = FALSE)
write.csv(Hold_sample,file = "Hold_Sample.csv",row.names = FALSE)

##########################################################################

#######################Feature engineering######################
Hold_sample1 = read.csv("Hold_Sample.csv")
Dev_sample1 = read.csv("Dev_Sample.csv")

time = strptime(Dev_sample1$ACC_OP_DATE,format = "%Y-%m-%d %H:%M:%S")
Dev_sample1$ACC_OP_DATE  = as.POSIXct(strptime(Dev_sample1$ACC_OP_DATE,"%m/%d/%Y"))
Dev_sample1$day <-  strftime(Dev_sample1$ACC_OP_DATE, '%u')
Dev_sample1$month = strftime(Dev_sample1$ACC_OP_DATE, '%m')
Dev_sample1$year = strftime(Dev_sample1$ACC_OP_DATE, '%Y')


time = strptime(Hold_sample1$ACC_OP_DATE,format = "%Y-%m-%d %H:%M:%S")
Hold_sample1$ACC_OP_DATE  = as.POSIXct(strptime(Hold_sample1$ACC_OP_DATE,"%m/%d/%Y"))
Hold_sample1$day <-  strftime(Hold_sample1$ACC_OP_DATE, '%u')
Hold_sample1$month = strftime(Hold_sample1$ACC_OP_DATE, '%m')
Hold_sample1$year = strftime(Hold_sample1$ACC_OP_DATE, '%Y')

#################################################################################

############################  Model creation###################################
forest = randomForest(as.factor(TARGET)~ . ,data = Dev_sample1[,c(-1,-11,-40)],ntree=101, mtry = 6 , nodesize = 15,importance=TRUE)
plot(forest)

## deciling code


Dev_sample1$predict.class <- predict(forest, Dev_sample1, type="class")
Dev_sample1$predict.score <- predict(forest, Dev_sample1, type="prob")

decile <- function(x){
  deciles <- vector(length=10)
  for (i in seq(0.1,1,.1)){
    deciles[i*10] <- quantile(x, i, na.rm=T)
  }
  return (
    ifelse(x<deciles[1], 1,
           ifelse(x<deciles[2], 2,
                  ifelse(x<deciles[3], 3,
                         ifelse(x<deciles[4], 4,
                                ifelse(x<deciles[5], 5,
                                       ifelse(x<deciles[6], 6,
                                              ifelse(x<deciles[7], 7,
                                                     ifelse(x<deciles[8], 8,
                                                            ifelse(x<deciles[9], 9, 10
                                                            ))))))))))
}
Dev_sample1$deciles <- decile(Dev_sample1$predict.score[,2])

summary(as.factor(Dev_sample1$TARGET))

## deciling



tmp_DT = data.table(Dev_sample1)
rank <- tmp_DT[, list(
  cnt = length(TARGET), 
  cnt_resp = sum(TARGET), 
  cnt_non_resp = sum(TARGET == 0)) , 
  by=deciles][order(-deciles)]
rank$rrate <- round (rank$cnt_resp / rank$cnt,2);
rank$cum_resp <- cumsum(rank$cnt_resp)
rank$cum_non_resp <- cumsum(rank$cnt_non_resp)
rank$cum_rel_resp <- round(rank$cum_resp / sum(rank$cnt_resp),2);
rank$cum_rel_non_resp <- round(rank$cum_non_resp / sum(rank$cnt_non_resp),2);
rank$ks <- abs(rank$cum_rel_resp - rank$cum_rel_non_resp);


rank$rrate <- percent(rank$rrate)
rank$cum_rel_resp <- percent(rank$cum_rel_resp)
rank$cum_rel_non_resp <- percent(rank$cum_rel_non_resp)

View(rank)

###################################################################################

######################Hold out data prediction####################################


Hold_sample1$predict.class <- predict(forest, Hold_sample1, type="class")
Hold_sample1$predict.score <- predict(forest, Hold_sample1, type="prob")
Hold_sample1$deciles <- decile(Hold_sample1$predict.score[,2])


tmp_DT = data.table(Hold_sample1)
h_rank <- tmp_DT[, list(
  cnt = length(TARGET), 
  cnt_resp = sum(TARGET), 
  cnt_non_resp = sum(TARGET == 0)) , 
  by=deciles][order(-deciles)]
h_rank$rrate <- round (h_rank$cnt_resp / h_rank$cnt,2);
h_rank$cum_resp <- cumsum(h_rank$cnt_resp)
h_rank$cum_non_resp <- cumsum(h_rank$cnt_non_resp)
h_rank$cum_rel_resp <- round(h_rank$cum_resp / sum(h_rank$cnt_resp),2);
h_rank$cum_rel_non_resp <- round(h_rank$cum_non_resp / sum(h_rank$cnt_non_resp),2);
h_rank$ks <- abs(h_rank$cum_rel_resp - h_rank$cum_rel_non_resp);


library(scales)
h_rank$rrate <- percent(h_rank$rrate)
h_rank$cum_rel_resp <- percent(h_rank$cum_rel_resp)
h_rank$cum_rel_non_resp <- percent(h_rank$cum_rel_non_resp)

View(h_rank)
########################################################################################

######################################Other performance parameters#######################
library(ROCR)
pred <- prediction(Dev_sample1$predict.score[,2], Dev_sample1$TARGET)
perf <- performance(pred, "tpr", "fpr")
plot(perf)
KS <- max(attr(perf, 'y.values')[[1]]-attr(perf, 'x.values')[[1]])
KS

## Area Under Curve
auc <- performance(pred,"auc"); 
auc <- as.numeric(auc@y.values)
auc

## Gini Coefficient
library(ineq)
gini = ineq(Dev_sample1$predict.score[,2], type="Gini")
gini

## Classification Error
with(Dev_sample1, table(TARGET, predict.class))

##################################################################################


