library(ggplot2)
library(caret)
library(parallel)
library(doParallel)
library(rpart)
library(rpart.plot)
library(data.table)
library(RWeka)
library(class)
library(tm)

file <- file.path(getwd(), "C:/Users/Alessandro/Documents/R/machine/pml-training.csv")
data_set_training<- read.csv(file,sep=",")
file <- file.path(getwd(), "R/machine/pml-testing.csv")
data_set_testing<- data.table(read.csv(file,sep=","))

data_set_training <- data_set_training[c("user_name", "new_window", "num_window", "roll_belt", "pitch_belt", "yaw_belt",        
                     "total_accel_belt", "gyros_belt_x", "gyros_belt_y", "gyros_belt_z", "accel_belt_x",   
                     "accel_belt_y", "accel_belt_z", "magnet_belt_x", "magnet_belt_y", "magnet_belt_z",    
                     "roll_arm", "pitch_arm", "yaw_arm", "total_accel_arm", "gyros_arm_x", "gyros_arm_y",  
                     "gyros_arm_z", "accel_arm_x", "accel_arm_y", "accel_arm_z", "magnet_arm_x",   
                     "magnet_arm_y", "magnet_arm_z", "roll_dumbbell", "pitch_dumbbell", "yaw_dumbbell",
                     "total_accel_dumbbell", "gyros_dumbbell_x", "gyros_dumbbell_y", "gyros_dumbbell_z",   
                     "accel_dumbbell_x", "accel_dumbbell_y", "accel_dumbbell_z", "magnet_dumbbell_x",  
                     "magnet_dumbbell_y", "magnet_dumbbell_z", "roll_forearm", "pitch_forearm", "yaw_forearm",
                     "total_accel_forearm", "gyros_forearm_x", "gyros_forearm_y", "gyros_forearm_z", "accel_forearm_x",    
                     "accel_forearm_y", "accel_forearm_z", "magnet_forearm_x", "magnet_forearm_y", "magnet_forearm_z", "classe")]


str(data_set_testing)
summary(data_set_training$classe)

table(data_set_training$classe)
data_set_training$classe<-factor(data_set_training$classe,levels=c("A","B","C","D","E"),labels=c("A","B","C","D","E"))
round(prop.table(table(data_set_training$classe))*100,digits=1)


class_training<-rpart(classe ~ .,data=training,method="class" )
rpart.plot(class_training,main="Decision tree",digits=2)

tree_training<-predict(class_training,testing,type="class")
confusionMatrix(tree_training,testing$classe)


set.seed(4600)
##Create a random partitioning using the caret package
inTrain = createDataPartition(data_set_training$classe, p = 3/4)[[1]]
## create a training dataset with 75% random data from the original dataset to build a suitable model
training <- data_set_training[inTrain,]
## Create a testing dataset with 25% random data from the original dataset to test the model
testing <- data_set_training[-inTrain,]

cl <- makeCluster(detectCores() - 3)
registerDoParallel(cl, cores = detectCores() - 3)
p_training<-randomForest(classe ~ .,data=training)
rf_predict<-predict(p_training,testing,type="class")
confusionMatrix(rf_predict,testing$classe)