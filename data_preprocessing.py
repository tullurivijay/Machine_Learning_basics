# Data Preprocessing Template

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Data.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 3].values


# Taking care of Missing Data
from sklearn.impute import SimpleImputer
imp_mean = SimpleImputer(missing_values= np.nan, strategy='mean')
imp_mean.fit(X[:,1:3])
X[:,1:3]= imp_mean.transform(X[:,1:3])
print(X[:,1:3])

# Encoding Categorical Data


# This is encoding for countries becuse machine learning models take only number but not categories that's
#reason why we are encoding, as we see in the output, we have again one more problem France - 0, Spain - 2
# Germany - 1, where it is considered as 0<1<2 that is not the case here because they are countries but not
# example measurements, sizes like that so we need DUMMY ENCODING THAT IS ONEHOTENCODING THAT SPLITTING THESE INTO 3 COLUMNS AND 
# REPRESENTED AS 1 AND 0'S LIKE IF WE ARE IN FRANCE THEN THAT IS 1 AND REMAINING ARE ZEROS

from sklearn.preprocessing import LabelEncoder
LabelEncoder_X = LabelEncoder()
X[:,0]= LabelEncoder_X.fit_transform(X[:,0])

# SO INSTEAD OF USING ABOVE 3 LINES WE ARE USING BELOW CODE


from sklearn.preprocessing import OneHotEncoder

onehotencoder_x = OneHotEncoder(categorical_features=[0])
X = onehotencoder_x.fit_transform(X).toarray()

# As Y is the dependent or target variable, ML model automativcaly know that this is category and need not to do One hot encoding
# It is go to with normal encoding process that is by label Encoder, which is need not to split into columns

LabelEncoder_Y =  LabelEncoder()
y = LabelEncoder_Y.fit_transform(y)


# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)


# As all the variables will be on the same scale for example in this salaries and ages, they are not on same scale
# Because of that when we are plotting a graph, the ages will be ignored because age 32 is not on the scale of salary 72000
# that's we need a transformation in order to maintain same scale
# This process is done by STANDARDIZATION AND NORMALIZATION --> Xstad = ((x - mean(x))/std(x)), Xnorm= x-min(x)/ max - min
# Feature Scaling

from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
# for the train set we have to do fit and transform and where as we need to only transform as they are already fitted with the object
# When We are scaling these variables, One may ask do we need to scale dummy encoded varaibles because they are already on good scale
# It depends on context, for example here if we scale country variable, we cant tell which observation belongs to which country
# But scaling those encoded variables gives us better predictions and accuray and mostly will between -1 and +1
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
# Here we are not scaling the dependent variable as it is on scale of 0 and 1, mostly this categorical dependent
