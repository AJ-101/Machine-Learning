# -*- coding: utf-8 -*-
"""
Created on Wed Mar 11 14:53:37 2020

@author: AJ
"""

#import libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#import dataset
dataset_train = pd.read_csv('train.csv')
X_train = dataset_train.iloc[:, [2,4,5,6,7,9,10,11]].values
y_train = dataset_train.iloc[:, 1].values

dataset_test = pd.read_csv('test.csv')
X_test = dataset_test.iloc[:, [1,3,4,5,6,8,9,10]].values

# Taking care of missing data
from sklearn.impute import SimpleImputer
imputer_train = SimpleImputer(strategy = 'mean')
imputer_train = imputer_train.fit(X_train[:, 2:3])
X_train[:, 2:3] =  imputer_train.transform(X_train[:, 2:3])
imputer1_train = SimpleImputer(strategy = "most_frequent")
imputer1_train = imputer1_train.fit(X_train[:, 6:8])
X_train[:, 6:8] =  imputer1_train.transform(X_train[:, 6:8])

imputer_test = SimpleImputer(strategy = 'mean')
imputer_test = imputer_test.fit(X_test[:, 2:6])
X_test[:, 2:6] =  imputer_test.transform(X_test[:, 2:6])
imputer1_test = SimpleImputer(strategy = "most_frequent")
imputer1_test = imputer1_test.fit(X_test[:, 6:8])
X_test[:, 6:8] =  imputer1_test.transform(X_test[:, 6:8])

# Encoding categorical data
# Encoding the Independent Variable
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X_train = LabelEncoder()
X_train[:, 1] = labelencoder_X_train.fit_transform(X_train[:, 1])
labelencoder_X1_train = LabelEncoder()
X_train[:, 6] = labelencoder_X1_train.fit_transform(X_train[:, 6])
labelencoder_X2_train = LabelEncoder()
X_train[:, 7] = labelencoder_X2_train.fit_transform(X_train[:, 7])
#Dummy Variables
onehotencoder = OneHotEncoder(categorical_features = [6])
X_train = onehotencoder.fit_transform(X_train).toarray()
X_train = X_train[:, 1:]
onehotencoder1 = OneHotEncoder(categorical_features = [152])
X_train = onehotencoder1.fit_transform(X_train).toarray()
X_train = X_train[:, 1:]

labelencoder_X_test = LabelEncoder()
X_test[:, 1] = labelencoder_X_test.fit_transform(X_test[:, 1])
labelencoder_X1_test = LabelEncoder()
X_test[:, 6] = labelencoder_X1_test.fit_transform(X_test[:, 6])
labelencoder_X2_test = LabelEncoder()
X_test[:, 7] = labelencoder_X2_test.fit_transform(X_test[:, 7])

X_train = X_train[:, 71:]
#Dummy Variables
onehotencoder_test = OneHotEncoder(categorical_features = [6])
X_test = onehotencoder_test.fit_transform(X_test).toarray()
X_test = X_test[:, 1:]
onehotencoder1_test = OneHotEncoder(categorical_features = [81])
X_test = onehotencoder1_test.fit_transform(X_test).toarray()
X_test = X_test[:, 1:]

#Fitting Random Forest to training set
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators = 1000, criterion = 'gini', random_state = 0)
classifier.fit(X_train, y_train)

#Testing predictions
y_pred = classifier.predict(X_test)

#Validation
from sklearn.model_selection import cross_val_score
accuracy = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = 10)
accuracy.mean()
