set.seed(1)
x<-runif(10,min = 0, max = 1)
abalone.df<-read.csv("abalone.data", header = F, sep = ",")
colnames(abalone.df) <- c("Sex","Length","Diam","Height","Whole","Shucked","Viscera","Shell","Rings")
options(digits = 3)
library(pastecs)
stat.desc(abalone.df)
library(GGally)
ggpairs(abalone.df[c(2,3,4,5,6,7,8,9)])
abalonemod1<-lm(Rings~Sex+Length+Diam+Height+Whole+Shucked+Viscera+Shell, data = abalone.df)
summary(abalonemod1)
library(car)
vif(abalonemod1)
par(mfrow=c(3,2))
plot(abalonemod1, which = 1:5)
shapiro.test(abalonemod1$residuals)
abalonemod2<-lm(Rings~Sex+Length+Diam+Height+Shucked+Viscera+Shell, data = abalone.df)
summary(abalonemod2)
vif(abalonemod2)
abalonemod3<-lm(Rings~Sex+Length+Height+Shucked+Viscera+Shell, data = abalone.df)
summary(abalonemod3)
vif(abalonemod3)
abalonemod4<-lm(Rings~Sex+Length+Height+Shucked+Shell, data = abalone.df)
summary(abalonemod4)
vif(abalonemod4)
#Threshold of VIF for Sex due to being a qualitative variable is the following 10^(1/(2*Df)). Df = 2, Threshold = 1.78
par(mfrow=c(3,2))
plot(abalonemod4, which = 1:5)
shapiro.test(abalonemod4$residuals)
library(MASS)
abalonemod5<-boxcox((Rings~Sex+Length+Height+Shucked+Shell), data=abalone.df, lambda=seq(from=-1, to=1, by=0.1))
cooksd<-cooks.distance(abalonemod4)
outliers<-which(cooksd > (4/ length(cookd)))
print(outliers)
abalone_r_outlier<-abalone.df[-outliers, ]
abalonemod6<-lm(Rings~Sex+Length+Height+Shucked+Shell, data = abalone_r_outlier)
summary(abalonemod6)
par(mfrow=c(3,2))
plot(abalonemod6, which = 1:5)
shapiro.test(abalonemod6$residuals)

wine.df<-read.csv("wine.data", header = F, sep = ",")
colnames(wine.df) <- c("Wine_Class","Alcohol","Malic_Acid","Ash","Alkalinity_of_Ash","Magnesium","Total_Phenols",
                       "Flavanoids","Nonflavanoid_Phenols","Proanthocyanins","Color_Intensity","Hue","OD280_OD315",
                       "Proline")
summary(wine.df)
library(pastecs)
stat.desc(wine.df)
library(GGally)
ggpairs(wine.df[c(2,3,4,5,6,7,8,9,10,11,12,13,1)])
wine.df$`Wine_Class`<-as.factor(wine.df$`Wine_Class`)
set.seed(1)
library(caTools)
split<-sample.split(wine.df$`Wine_Class`, SplitRatio = 0.7)
train_set<-subset(wine.df, split ==TRUE)
test_set<-subset(wine.df, split == FALSE)
library(nnet)
winemod1 <- multinom(Wine_Class ~ ., data = train_set)
summary(winemod1)
train_p_class <- predict(winemod1, train_set)
train_accuracy <- mean(train_p_class == train_set$Wine_Class)
print(paste("Training Accuracy:", round(train_accuracy * 100, 2), "%"))
test_p_class <- predict(winemod1, test_set)
test_accuracy <- mean(test_p_class == test_set$Wine_Class)
print(paste("Test Accuracy:", round(test_accuracy * 100, 2), "%"))
test_p_class <- factor(test_p_class, levels = levels(test_set$Wine_Class))
library(caret)
conf_matrix <- confusionMatrix(test_p_class, test_set$Wine_Class)
print(conf_matrix)
library(dplyr)
wine_scaled<-wine.df %>%
  mutate(across(-Wine_Class, scale))
X_train <- train_set[, -1]  # Exclude Wine_Class
y_train <- train_set$Wine_Class
X_test <- test_set[, -1]
y_test <- test_set$Wine_Class
# Define a range for K
k_values <- seq(1, 30, by = 2)  # Test odd values from 1 to 30
train_accuracies <- c()
test_accuracies <- c()
library(class)
library(caret)
library(ggplot2)
library(dplyr)

for (k in k_values) {
  knn_pred_train <- knn(train = X_train, test = X_train, cl = y_train, k = k)
  knn_pred_test <- knn(train = X_train, test = X_test, cl = y_train, k = k)
  
  train_acc <- mean(knn_pred_train == y_train)
  test_acc <- mean(knn_pred_test == y_test)
  
  train_accuracies <- c(train_accuracies, train_acc)
  test_accuracies <- c(test_accuracies, test_acc)
}


optimal_k <- k_values[which.max(test_accuracies)]
print(paste("Optimal K:", optimal_k))


accuracy_df <- data.frame(k_values, train_accuracies, test_accuracies)

ggplot(accuracy_df, aes(x = k_values)) +
  geom_line(aes(y = train_accuracies, color = "Training Accuracy")) +
  geom_line(aes(y = test_accuracies, color = "Testing Accuracy")) +
  geom_vline(xintercept = 9, linetype = "dashed", color = "red") +
  labs(title = "KNN Accuracy vs. K", x = "Number of Neighbors (K)", y = "Accuracy") +
  theme_minimal() +
  scale_color_manual(values = c("blue", "orange"))

final_knn_pred_train <- knn(train = X_train, test = X_train, cl = y_train, k = 9)
final_knn_pred_test <- knn(train = X_train, test = X_test, cl = y_train, k = 9)

final_train_accuracy <- mean(final_knn_pred_train == y_train)
final_test_accuracy <- mean(final_knn_pred_test == y_test)

print(paste("Final Training Accuracy:", round(final_train_accuracy * 100, 2), "%"))
print(paste("Final Testing Accuracy:", round(final_test_accuracy * 100, 2), "%"))
conf_matrix <- confusionMatrix(final_knn_pred_test, y_test)
print(conf_matrix)

