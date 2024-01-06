#Home Exercise 2----
# Alsiraira Mokhtar
# prediction of bike rental 

# task 1----
#In this exercise, you are asked to develop 
#regression algorithms to predict how many bikes will be rented within each slot. 

#install the neededd libraries
install.packages("readr")
install.packages("e1071")
install.packages("rpart.plot")
install.packages("vcd")
install.packages("mgcv")
install.packages("factoextra")
install.packages("ggtext")

# Load required libraries
library(dplyr)
library(ggplot2)
library(readr)
library(knitr)
library(car)
library(lubridate)
library(caret)
library(e1071)
library(rpart)
library(rpart.plot)
library(vcd)
library(mgcv)
library(randomForest)
library(tibble)
library(cluster)
library(factoextra)


# Read the train dataset
train_data <- read_csv("train.csv")

# Read the test dataset
test_data <- read_csv("test.csv")


# Display the first few rows of the train dataset
head(train_data)


# Display the first few rows of the test dataset
head(test_data)

# Preprocessing the data sets
# Convert datetime column to character format
train_data$datetime <- as.character(train_data$datetime)

test_data$datetime <- as.character(test_data$datetime)

# Define multiple format options for parsing datetime column
date_formats <- c("%Y-%m-%d %H:%M:%S")


# Separate datetime column into year, month, day, and hour for train dataset
train_data$datetime <- parse_datetime(train_data$datetime, format = date_formats)

train_data$year <- year(train_data$datetime)
train_data$month <- month(train_data$datetime)
train_data$day <- day(train_data$datetime)
train_data$hour <- hour(train_data$datetime)

train_data$month <- factor(train_data$month)
train_data$hour <- factor(train_data$hour)
train_data$day <- factor(train_data$day)


# Convert variables to factors or character vectors
train_data$season <- as.factor(train_data$season)
train_data$holiday <- as.factor(train_data$holiday)
train_data$workingday <- as.factor(train_data$workingday)
train_data$weather <- as.factor(train_data$weather)

# Convert year variable to factor
train_data$year <- factor(train_data$year)

# Convert the datetime column to a proper date-time format
train_data$datetime <- as.POSIXct(train_data$datetime, format = "%m/%d/%Y %I:%M:%S %p")

# Extract the weekday from the datetime column
train_data$weekday <- weekdays(train_data$datetime)

# Convert the weekday to a factor variable
train_data$weekday <- factor(train_data$weekday, levels = c("Sunday", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"))


# Convert the weekday column to numeric values from 1 to 7
train_data$weekday <- wday(train_data$datetime, label = FALSE)


# Separate datetime column into year, month, day, and hour for test dataset
test_data$datetime <- parse_datetime(test_data$datetime, format = date_formats)

test_data$year <- year(test_data$datetime)
test_data$month <- month(test_data$datetime)
test_data$day <- day(test_data$datetime)
test_data$hour <- hour(test_data$datetime)

test_data$month <- factor(test_data$month)
test_data$hour <- factor(test_data$hour)
test_data$day <- factor(test_data$day)


test_data$season <- as.factor(test_data$season)
test_data$holiday <- as.factor(test_data$holiday)
test_data$workingday <- as.factor(test_data$workingday)
test_data$weather <- as.factor(test_data$weather)

test_data$year <- factor(test_data$year)

# Display the updated train dataset with separated columns
head(train_data)

# Display the updated test dataset with separated columns
head(test_data)

# Get the levels of 'weather' variable in the training data
weather_levels <- levels(train_data$weather)

# Modify the levels of 'weather' variable in the test data to match the training data
test_data$weather <- factor(test_data$weather, levels = weather_levels)

# check the structure of the data
str(train_data)
str(test_data)

# Update factor variables with labels
train_data$season <- factor(train_data$season, levels = c(1, 2, 3, 4), labels = c("spring", "summer", "fall", "winter"))
train_data$holiday <- factor(train_data$holiday, levels = c(0, 1), labels = c("no", "yes"))
train_data$workingday <- factor(train_data$workingday, levels = c(0, 1), labels = c("no", "yes"))
train_data$weather <- factor(train_data$weather, levels = c(1, 2, 3, 4), labels = c("clear", "mist", "light", "heavy"))

test_data$season <- factor(test_data$season, levels = c(1, 2, 3, 4), labels = c("spring", "summer", "fall", "winter"))
test_data$holiday <- factor(test_data$holiday, levels = c(0, 1), labels = c("no", "yes"))
test_data$workingday <- factor(test_data$workingday, levels = c(0, 1), labels = c("no", "yes"))
test_data$weather <- factor(test_data$weather, levels = c(1, 2, 3, 4), labels = c("clear", "mist", "light", "heavy"))


# Convert the datetime column in the test data to a proper date-time format
test_data$datetime <- as.POSIXct(test_data$datetime, format = "%m/%d/%Y %I:%M:%S %p")

# Extract the weekday from the datetime column
test_data$weekday <- weekdays(test_data$datetime)

# Convert the weekday to a factor variable
test_data$weekday <- factor(test_data$weekday, levels = c("Sunday", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"))


# Convert the weekday column to numeric values from 1 to 7
test_data$weekday <- wday(test_data$datetime, label = FALSE)

# Print the updated data with the numeric weekday column
head(train_data)


# Print the updated data with the weekday column
head(test_data)

#check the structure of the data
str(train_data)

str(test_data)


# Handling missing values
# Check for missing values in the dataset
missing_values <- sum(is.na(train_data))

if (missing_values > 0) {
  print(paste("Number of missing values:", missing_values))
} else {
  print("No missing values found.")
}


# EDA on training data----
# Summary statistics
# Data Exploration

summary(train_data)

# Distribution of variables

# Plot histogram with color
ggplot(train_data, aes(x = count, fill = (count))) +
  geom_histogram(binwidth = 100, color = "blue") +
  scale_fill_gradient(low = "blue", high = "red") +
  labs(title = "Distribution of Bike Counts", x = "Count", y = "Frequency")

# Scatter plot matrix
pairs(train_data[, c("count", "season", "holiday", "weather", "temp")])

# Histogram of bike counts
ggplot(train_data, aes(x = count)) +
  geom_histogram(fill = "steelblue", color = "white") +
  labs(title = "Distribution of Bike Counts",
       x = "Count", y = "Frequency")

# Boxplot of bike counts by season
ggplot(train_data, aes(x = season, y = count, fill = season)) +
  geom_boxplot() +
  labs(title = "Bike Counts by Season",
       x = "Season", y = "Count")

# Plot casual and registered bike counts per holiday
ggplot(train_data, aes(x = holiday, y = casual, fill = holiday)) +
  geom_bar(stat = "identity", position = "dodge") +
  labs(x = "Holiday", y = "Casual Count", fill = "Holiday") +
  scale_fill_manual(values = c("#F8766D", "#00BFC4"), labels = c("No", "Yes")) +
  ggtitle("Casual Bike Counts per Holiday") +
  theme_minimal()

# Plot casual and registered bike counts per holiday
ggplot(train_data, aes(x = holiday, y = registered, fill = holiday)) +
  geom_bar(stat = "identity", position = "dodge") +
  labs(x = "Holiday", y = "Registered Count", fill = "Holiday") +
  scale_fill_manual(values = c("#F8766D", "#00BFC4"), labels = c("No", "Yes")) +
  ggtitle("Registered Bike Counts per Holiday") +
  theme_minimal()


# Plot casual and registered bike counts per season
ggplot(train_data, aes(x = season, y = casual, fill = season)) +
  geom_bar(stat = "identity", position = "dodge") +
  labs(x = "Season", y = "Casual Count", fill = "Season") +
  ggtitle("Casual Bike Counts per Season") +
  theme_minimal()

# Plot casual and registered bike counts per season
ggplot(train_data, aes(x = season, y = registered, fill = season)) +
  geom_bar(stat = "identity", position = "dodge") +
  labs(x = "Season", y = "Registered Count", fill = "Season") +
  ggtitle("Registered Bike Counts per Season") +
  theme_minimal()

# Plot casual and registered bike counts per weather
ggplot(train_data, aes(x = weather, y = casual, fill = weather)) +
  geom_bar(stat = "identity", position = "dodge") +
  labs(x = "Weather", y = "Casual Count", fill = "Weather") +
  ggtitle("Casual Bike Counts per Weather") +
  theme_minimal()

# Plot casual and registered bike counts per weather
ggplot(train_data, aes(x = weather, y = registered, fill = weather)) +
  geom_bar(stat = "identity", position = "dodge") +
  labs(x = "Weather", y = "Registered Count", fill = "Weather") +
  ggtitle("Registered Bike Counts per Weather") +
  theme_minimal()

# Plot casual and registered bike counts per year
ggplot(train_data, aes(x = year, y = casual, fill = year)) +
  geom_bar(stat = "identity", position = "dodge") +
  labs(x = "Year", y = "Casual Count", fill = "Year") +
  scale_fill_manual(values = c("#F8766D", "#00BFC4"), labels = c("2011", "2012")) +
  ggtitle("Casual Bike Counts per Year") +
  theme_minimal()

# Plot casual and registered bike counts per year
ggplot(train_data, aes(x = year, y = registered, fill = year)) +
  geom_bar(stat = "identity", position = "dodge") +
  labs(x = "Year", y = "Registered Count", fill = "Year") +
  scale_fill_manual(values = c("#F8766D", "#00BFC4"), labels = c("2011", "2012")) +
  ggtitle("Registered Bike Counts per Year") +
  theme_minimal()

# Bar plot of bike counts by weather situation
ggplot(train_data, aes(x = weather, fill = weather)) +
  geom_bar() +
  labs(title = "Bike Counts by Weather Situation",
       x = "Weather", y = "Count")

# Correlation matrix
cor_matrix <- cor(train_data[, c("temp", "atemp", "humidity", "windspeed", "casual", "registered", "count")])
cor_matrix

# Convert correlation matrix to tidy data frame
cor_df <- as.data.frame(as.table(cor_matrix))
colnames(cor_df) <- c("Variable1", "Variable2", "Correlation")

# Create correlation heatmap
ggplot(cor_df, aes(x = Variable1, y = Variable2, fill = Correlation)) +
  geom_tile() +
  scale_fill_gradient2(low = "blue", high = "red", midpoint = 0, limits = c(-1, 1)) +
  theme(axis.text.x = element_text(angle = 45, hjust = 1)) +
  labs(title = "Correlation Heatmap")
# from the correlation matrix we notice that temp and atemp are highly correlated(.9)

# Create boxplots for numerical variables
boxplot(train_data[, c("temp", "humidity", "windspeed")])

# Calculate summary statistics for numerical variables
summary(train_data[, c("temp", "humidity", "windspeed")])

#from the boxplot it is clear that windspeed and humidity have outliers
# replace the outliers with mean

# Calculate the mean value of each variable
mean_humidity <- mean(train_data$humidity)
mean_windspeed <- mean(train_data$windspeed)

# Define the threshold for outliers (e.g., more than 3 standard deviations away from the mean)
threshold <- 3

# Identify and replace outliers in humidity
outliers_humidity <- train_data$humidity > (mean_humidity + threshold * sd(train_data$humidity))

# Calculate outliers for humidity
humidity_outliers <- boxplot.stats(train_data$humidity)$out
humidity_outliers_count <- length(humidity_outliers)
cat("Number of outliers in humidity:", humidity_outliers_count, "\n")

# Calculate outliers for windspeed
windspeed_outliers <- boxplot.stats(train_data$windspeed)$out
windspeed_outliers_count <- length(windspeed_outliers)
cat("Number of outliers in windspeed:", windspeed_outliers_count, "\n")

train_data$humidity[outliers_humidity] <- mean_humidity

# Identify and replace outliers in windspeed
outliers_windspeed <- train_data$windspeed > (mean_windspeed + threshold * sd(train_data$windspeed))
train_data$windspeed[outliers_windspeed] <- mean_windspeed

#check which variables are correlated 

# Calculate correlation matrix for numeric variables
numeric_vars <- c("temp", "humidity", "windspeed")
numeric_cor <- cor(train_data[numeric_vars])

# Identify highly correlated numeric variables
numeric_cor_threshold <- 0.8
highly_correlated_numeric <- which(abs(numeric_cor) > numeric_cor_threshold, arr.ind = TRUE)
highly_correlated_numeric <- unique(highly_correlated_numeric[, "row"])# Calculate correlation between temperature and wind speed
corr_temp_wind <- cor(train_data$temp, train_data$windspeed)

# Calculate correlation between temperature and humidity
corr_temp_humidity <- cor(train_data$temp, train_data$humidity)

# Compare the correlation coefficients
if (abs(corr_temp_wind) > abs(corr_temp_humidity)) {
  print("Wind speed is more strongly correlated with temperature.")
} else if (abs(corr_temp_wind) < abs(corr_temp_humidity)) {
  print("Humidity is more strongly correlated with temperature.")
} else {
  print("Both wind speed and humidity have similar correlation with temperature.")
}

# Calculate Cramér's V for categorical variables
categorical_vars <- c("season", "weather","month","hour","day","holiday","year","workingday","weekday")
categorical_cramer_v <- matrix(0, nrow = length(categorical_vars), ncol = length(categorical_vars), dimnames = list(categorical_vars, categorical_vars))

for (i in 1:length(categorical_vars)) {
  for (j in 1:length(categorical_vars)) {
    if (i == j) {
      categorical_cramer_v[i, j] <- 1
    } else {
      cross_tab <- table(train_data[[categorical_vars[i]]], train_data[[categorical_vars[j]]])
      categorical_cramer_v[i, j] <- assocstats(cross_tab)$cramer
    }
  }
}

# Set a threshold for high correlation
categorical_cor_threshold <- 0.8

# Identify highly correlated categorical variables
highly_correlated_categorical <- which(abs(categorical_cramer_v) > categorical_cor_threshold, arr.ind = TRUE)
highly_correlated_categorical <- unique(highly_correlated_categorical[, "row"])

# Combine highly correlated variables
highly_correlated_vars <- highly_correlated_categorical

highly_correlated_var_names <- categorical_vars[highly_correlated_vars]

highly_correlated_var_names 

# the correlation analysis indicate that the Humidity is more strongly correlated with temperature
# but when removing it from the model the R squared decreasea alot.

# prepare the dummy variable for the model----

new_train_data <- train_data[]


new_train_data <- subset(new_train_data, select = -c(...1, datetime,casual, registered,atemp))
variable.names(new_train_data)


# Specify the categorical variables
categorical_vars <- c("season","month","year","holiday", "workingday","weather","hour","weekday","day")

# Create dummy variables
dummies <- dummyVars(~., data = new_train_data[, categorical_vars], fullRank = TRUE)

dummy_data <- predict(dummies, newdata = new_train_data[, categorical_vars])


# Combine dummy variables with the remaining variables
new_train_data <- cbind(dummy_data, new_train_data[, -which(names(new_train_data) %in% categorical_vars)])


# linear regression----
# Fit linear regression model
model_lm <- lm(count ~ ., data = new_train_data)


# Print model summary
summary(model_lm)


# Assuming you have fitted a linear regression model named "lm_model" on your training data

# Make predictions on the training data
lm_train_predictions <- predict(model_lm, new_train_data)

# count how many negative values in the predictions
sum(lm_train_predictions < 0)

#convert negative predictions in lm_train_predictions to 0 using the pmax()
lm_train_predictions <- pmax(lm_train_predictions, 0)
# Calculate the RMSE
train_rmse_lm <- sqrt(mean((lm_train_predictions - new_train_data$count)^2))

# Print the RMSE
cat("Training RMSE:", train_rmse_lm, "\n")


# Perform backward selection
model_backward_lm <- step(model_lm, direction = "backward")

# Print the selected model summary
summary(model_backward_lm)

# 
#prepare the test data----
new_test_data <- test_data[]

new_test_data <- subset(new_test_data, select = -c(...1, datetime,atemp))
variable.names(new_test_data)


# Specify the categorical variables
categorical_vars <- c("season", "holiday", "workingday", "weather", "year","day" ,"weekday","hour","month")

# Create dummy variables
dummies <- dummyVars(~., data = new_test_data[, categorical_vars], fullRank = TRUE)
dummy_data <- predict(dummies, newdata = new_test_data[, categorical_vars])

# Combine dummy variables with the remaining variables
new_test_data <- cbind(dummy_data, new_test_data[, -which(names(new_test_data) %in% categorical_vars)])


# Make predictions on the testing data
lm_test_predictions <- predict(model_lm, new_test_data)
lm_test_predictions
# count how many negative values in the predictions
negative_count <- sum(lm_test_predictions < 0)
negative_count

# SVM----

# Prepare the data
# Assuming your training dataset is stored in a data frame called "train_data"
# Separate the predictors (features) and the target variable (class)

SVM_train_data <- new_train_data[]

x_train <- SVM_train_data[, !(colnames(SVM_train_data) %in% c("count"))]
y_train <- SVM_train_data$count

# Train the SVM model
model_svm <- svm(x_train, y_train)
summary(model_svm)
# Make predictions on the training data
train_predictions_svm <- predict(model_svm, x_train)
# check how many predictions are negative
sum(train_predictions_svm < 0)

#convert negative predictions in train_predictions_svm to 0 using the pmax()
train_predictions_svm <- pmax(train_predictions_svm, 0)


# Evaluate the performance of the regression model (e.g., using RMSE or R-squared)
train_rmse_svm <- sqrt(mean((train_predictions_svm - y_train)^2))
cat("Training RMSE:", train_rmse_svm, "\n")

# apply SVM model on the test data
#prepare the test data----

new_test_data <- test_data[]

new_test_data <- subset(new_test_data, select = -c(...1, datetime,atemp))
variable.names(new_test_data)


# Specify the categorical variables
categorical_vars <- c("season", "holiday", "workingday", "weather", "year", "day","weekday","hour","month")

# Create dummy variables
dummies <- dummyVars(~., data = new_test_data[, categorical_vars], fullRank = TRUE)
dummy_data <- predict(dummies, newdata = new_test_data[, categorical_vars])

# Combine dummy variables with the remaining variables
new_test_data <- cbind(dummy_data, new_test_data[, -which(names(new_test_data) %in% categorical_vars)])

# test SVM

# Make predictions on the testing data

test_predictions_svm <- predict(model_svm, new_test_data)
test_predictions_svm 

#how many negative predictions
sum(test_predictions_svm < 0)


# decision tree----
# Assuming your train data is stored in a dataframe named "train_data"

# Specify the formula for the decision tree
formula <- count ~ .


# Train the decision tree model
model_dt <- rpart(formula, data = new_train_data)

# Print the decision tree model
print(model_dt)
# Plot the decision tree
rpart.plot(model_dt)

# Predict using the decision tree model
predictions_dt  <- predict(model_dt, newdata = new_train_data)

# View the predicted values
print(predictions_dt)

train_rmse_dt <- sqrt(mean((predictions_dt - new_train_data$count)^2))
cat("Training RMSE:", train_rmse_dt, "\n")

# Random forest 
# Specify the formula for the random forest model

rf_train_data<-new_train_data
formula <- count ~ .

# Train the random forest model
rf_model <- randomForest(formula, data = rf_train_data)

# Print the summary of the random forest model
print(rf_model)

# Predict using the random forest model
predictions_rf <- predict(rf_model, newdata = rf_train_data)

# View the predicted values
print(predictions_rf)
train_rmse_rf <- sqrt(mean((predictions_rf - train_data$count)^2))
cat("Training RMSE:", train_rmse_rf, "\n")

# plot the RMSE
# Create a vector of the RMSE values
rmse_values <- c(train_rmse_dt, train_rmse_svm, train_rmse_lm, train_rmse_rf)
model_labels <- c("Decision Tree", "SVM", "Linear Regression", "Random Forest")

# Define colors for the bars
bar_colors <- c("blue", "green", "orange", "red")

# Plot the RMSE values with centered labels and colors
barplot(rmse_values, names.arg = model_labels, ylab = "RMSE", main = "RMSE Comparison",
        col = bar_colors)

# Add the RMSE values as text inside the bars
text(x = barplot(rmse_values, names.arg = model_labels, ylab = "RMSE", main = "RMSE Comparison",
                 col = bar_colors, ylim = c(0, max(rmse_values) * 1.2)),
     y = rmse_values, label = round(rmse_values, digits = 2), pos = 3)


#10 fold cross validation for the three models
# Set the seed value
set.seed(123)  
# Define the training control for cross-validation
ctrl <- trainControl(method = "cv", number = 10)

#linear regression

# Perform cross-validation for linear regression
lm_results <- train(count ~ ., data = new_train_data, method = "lm", trControl = ctrl)
lm_results

#SVM

# Perform cross-validation for SVM
svm_results <- train(count ~ ., data = SVM_train_data, method = "svmLinear", trControl = ctrl)
svm_results
#Decision Tree

# Perform cross-validation for decision tree
dt_results <- train(count ~ ., data = new_train_data, method = "rpart", trControl = ctrl)
dt_results

# Create a table for linear regression results
lm_table <- tibble(
  Model = "Linear Regression",
  RMSE = lm_results$results$RMSE,
  R_squared = lm_results$results$Rsquared
)

# Create a table for SVM results
svm_table <- tibble(
  Model = "SVM",
  RMSE = svm_results$results$RMSE,
  R_squared = svm_results$results$Rsquared
)

# Create a table for decision tree results
dt_table <- tibble(
  Model = "Decision Tree",
  RMSE = dt_results$results$RMSE,
  R_squared = dt_results$results$Rsquared
)

# Print the individual tables
print(lm_table)
print(svm_table)
print(dt_table)

# Train the Random Forest model using cross-validation
#rf_results <- train(count ~ ., data = rf_train_data, method = "rf", trControl = ctrl)
#rf_results


# task 2----
#Do clustering in the dataset and 
#create clusters that separate the dataset into homogeneous groups.

cluster_test_data <-new_test_data[]

# Remove extra columns from the train dataset
cluster_train_data <-new_train_data[,-c(66)]
#cluster_train_data <-cluster_train_data[, !colnames(cluster_train_data) %in% c("registered", "casual","count")]

# Merge the train and test datasets

cluster_data <- rbind(cluster_train_data, cluster_test_data )

# Scale the continuous variables
scaled_data <- scale(cluster_data[, c("temp", "humidity", "windspeed")])

# Combine the scaled continuous variables with the one-hot encoded variables
dummy_data_cluster <- cbind(scaled_data, cluster_data[,-c(63,64,65)])


# Perform PCA
pca_result <- prcomp(dummy_data_cluster, scale = TRUE)

# Scree plot to determine the number of components
fviz_eig(pca_result, addlabels = TRUE)

# Elbow method to determine the optimal number of clusters
wss <- sapply(1:10, function(k) kmeans(pca_result$x, centers = k)$tot.withinss)
plot(1:10, wss, type = "b", xlab = "Number of Clusters", ylab = "Total Within Sum of Squares")

# Perform k-means clustering with the chosen number of clusters
k <- 5 # Change this value based on the elbow plot
kmeans_result <- kmeans(pca_result$x, centers = k)

# Visualize the clustering results
fviz_cluster(kmeans_result, data = pca_result$x)

# Extract cluster assignments
cluster_assignments <- kmeans_result$cluster

# Add cluster assignments to the original data frame
cluster_data$cluster <- cluster_assignments


summary_stats <- cluster_data %>%
  group_by(cluster) %>%
  summarise(
    mean_temp = mean(temp),
    min_temp = min(temp),
    max_temp = max(temp),
    mean_windspeed = mean(windspeed),
    min_windspeed = min(windspeed),
    max_windspeed = max(windspeed),
    mean_humidity = mean(humidity),
    min_humidity = min(humidity),
    max_humidity = max(humidity)
  )

# Print the summary statistics
print(summary_stats)


