# Data Preprocessing Template

# Importing the dataset
dataset = read.csv('Data.csv')

# Handling Missing Data
# In R Index starts with 1
dataset$Age = ifelse(is.na(dataset$Age),
                     ave(dataset$Age, FUN = function(x) mean(x,na.rm = TRUE)),
                     dataset$Age)

dataset$Salary = ifelse(is.na(dataset$Salary),
                     ave(dataset$Salary, FUN = function(x) mean(x,na.rm = TRUE)),
                     dataset$Salary)

# Encoding Variables using factor function by considering c as vector for the given column names
dataset$Country = factor(dataset$Country, 
                         levels = c('France', 'Spain','Germany'),
                         labels = c(1,2,3)
                         )

dataset$Purchased = factor(dataset$Purchased, 
                         levels = c('Yes', 'No'),
                         labels = c(0,1)
)


# Splitting the dataset into the Training set and Test set
#install.packages('caTools')
library(caTools)
# As we have selected the random state in the python in order to maintain same results on different platforms
set.seed(123)

# As we are single line in python, here we need to mention the dependent varaible as first parameter and split ratio for training set

split = sample.split(dataset$Purchased, SplitRatio = 0.8)
training_set = subset(dataset, split == TRUE)
test_set = subset(dataset, split == FALSE)

# Feature Scaling
# Here If we are exlcuding the categorical variables, because before we have transformed those categorical
# variables into factors above, which is not numeric in R but looks like numeric, so we have to exlcude
# categorical while scaling
training_set[,2:3]= scale(training_set[,2:3])
test_set[,2:3] = scale(test_set[,2:3])