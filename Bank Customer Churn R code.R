install.packages("tidyverse")
install.packages("dplyr")
install.packages("caret")
install.packages("tm")
install.packages("RColorBrewer")
install.packages("wordcloud")
install.packages("plyr")
install.packages("SnowballC")
install.packages("gmodels")
install.packages("forcats")
install.packages("corrplot")
install.packages("plotly")
install.packages("caTools")
install.packages("C50")
install.packages("ggcorrplot")
install.packages("DMwR")install.packages("e1071")
library(tidyverse) 
library(dplyr)
library(caret)
library(tm)
library(RColorBrewer)
library(wordcloud)
library(plyr)
library(SnowballC)
library(gmodels)
library(forcats)
library(corrplot)
library(plotly)
library(caTools)
library(C50)
library(ggcorrplot)
library(DMwR)
library(e1071)
library(ggiraph)
library(ggiraphExtra)
library(plyr)
data<-read.csv("C:\\Users\\Lenovo\\Desktop\\Projects\\Bank Customer Churn Model\\Data set\\Churn_Modelling.csv")
str(data)

#checking presence of NA in all our columns
colSums(is.na(data))

data<-data[-c(1,2,3)]
#to check if they are removed
str(data)
summary(data)

data$Exited<-mapvalues(data$Exited, from = c(0,1), to = c("no", "yes"))
data$IsActiveMember<-mapvalues(data$IsActiveMember, from = c(0,1), to = c("no", "yes"))
data$HasCrCard<-mapvalues(data$HasCrCard, from = c(0,1), to = c("no", "yes"))

#converting categorical variables to factors

data$Exited<-as.factor(data$Exited)
data$IsActiveMember<-as.factor(data$IsActiveMember)
data$HasCrCard<-as.factor(data$HasCrCard)
data$Geography<-as.factor(data$Geography)
data$Gender<-as.factor(data$Gender)
data$Tenure<-as.factor(data$Tenure)
data$NumOfProducts<-as.factor(data$NumOfProducts)

str(data)

#display the summary of descriptive statistics
summary(data)

#Exited = 1 non-churned customer
#Exited = 2 churned customer

ggplot(data = data, aes(x = Exited, fill =Geography )) + geom_bar()
ggplot(data = data, aes(x = Exited, fill =Gender )) + geom_bar()
ggplot(data = data, aes(x = HasCrCard, fill =Exited)) + geom_bar()
ggplot(data = data, aes(x = IsActiveMember, fill =Exited)) + geom_bar()
ggplot(data, aes(x = Exited, y = Age,fill=Exited)) + geom_boxplot()
ggplot(data, aes(x = Exited, y = EstimatedSalary,fill=Exited)) + geom_boxplot()
ggplot(data, aes(x = Exited, y = Balance,fill=Exited)) + geom_boxplot()
ggplot(data = data, aes(x = Exited ,fill=NumOfProducts)) + geom_bar()


ggplot(data, aes(Exited, fill = Exited)) + geom_bar() + theme(legend.position = 'none')

table(data$Exited)

round(prop.table(table(data$Exited)),3)

#overall distribution of every feature
#Categorical Variable Distribution
data %>%
  dplyr::select(-Exited) %>% 
  keep(is.factor) %>%
  gather() %>%
  group_by(key, value) %>% 
  dplyr::summarise(n = n()) %>% 
  ggplot() +
  geom_bar(mapping=aes(x = value, y = n, fill=key), color="black", stat='identity') + 
  coord_flip() +
  facet_wrap(~ key, scales = "free") +
  theme_minimal() +
  theme(legend.position = 'none')


#Continuous Variable Distribution
data %>%
  keep(is.numeric) %>%
  gather() %>%
  ggplot() +
  geom_histogram(mapping = aes(x=value,fill=key), color="black") +
  facet_wrap(~ key, scales = "free") +
  theme_minimal() +
  theme(legend.position = 'none')

#correlation Matrix
numericVarName <- names(which(sapply(data, is.numeric)))
corr <- cor(data[,numericVarName], use = 'pairwise.complete.obs')
ggcorrplot(corr, lab = TRUE)

 
#chi-Square test for feature selection 
chi.square <- vector()
p.value <- vector()
cateVar <- data %>% 
  dplyr::select(-Exited) %>% 
  keep(is.factor)

for (i in 1:length(cateVar)) {
  p.value[i] <- chisq.test(data$Exited, unname(unlist(cateVar[i])), correct = FALSE)[3]$p.value
  chi.square[i] <- unname(chisq.test(data$Exited, unname(unlist(cateVar[i])), correct = FALSE)[1]$statistic)
}

chi_sqaure_test <- tibble(variable = names(cateVar)) %>% 
  add_column(chi.square = chi.square) %>% 
  add_column(p.value = p.value)
knitr::kable(chi_sqaure_test)

#chi-Square values and p value of Tenure and HascrCard are very small so they are 
#not very significant for our churn prediction thus we can remove them 

data <- data %>% 
  dplyr::select(-Tenure, -HasCrCard)

#Now we will proceed with the prediction model 
#Pre Processing and data partition 

set.seed(1234)
sample_set <- data %>%
  pull(.) %>% 
  sample.split(SplitRatio = .7)

bankTrain <- subset(data, sample_set == TRUE)
bankTest <- subset(data, sample_set == FALSE)

round(prop.table(table(data$Exited)),3)

#Regression analysis
bankTrain <- SMOTE(Exited ~ ., data.frame(bankTrain), perc.over = 100, perc.under = 200)
round(prop.table(table(dplyr::select(bankTrain, Exited), exclude = NULL)),4)

## Train the model
logit.mod <- glm(Exited ~., family = binomial(link = 'logit'), data = bankTrain)

## Look at the result
summary(logit.mod)

## Predict the outcomes against our test data
logit.pred = rep("no",length(logit.pred.prob))
logit.pred[logit.pred.prob > 0.5] = "yes"
logit.pred <- as.factor(logit.pred)
confusionMatrix(logit.pred, bankTest$Exited, positive = "yes")
