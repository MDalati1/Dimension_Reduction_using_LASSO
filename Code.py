#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 29 09:28:22 2022

@author: mohamaddalati
"""
import pandas as pd 
import numpy as np
bank_df = pd.read_csv('/Users/mohamaddalati/Desktop/UniversalBank.csv')

###             Task 0: 
bank_df.drop(labels = 'UserID', axis = 1, inplace = True)
bank_df['Education'].value_counts() 

X = bank_df.drop(columns=['Personal Loan'])
y = bank_df['Personal Loan']


###             Task 1: 
"""
Perform feature selection process using LASSO with alpha = 0.05. Standardize all predictors using Z-score before
"""
# Standardize predictors using Z 
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_std = pd.DataFrame(X_scaled, columns = X.columns)
from sklearn.linear_model import Lasso 
lasso = Lasso(alpha = 0.05)
model = lasso.fit(X_std, y)
model.coef_
pd.DataFrame(list(zip(X.columns,model.coef_)), columns = ['predictor','coefficient'])
# Interpretation: Three variables; 'Income', 'CD Account' and 'Education'

###             Task 2:
""" 
Perform Dimension Reduction using PCA where the # of components is the # of features selected by LASSO observed above


NOTE: I will perform PCA WITH & WITHOUT STANDARDIZATION and pick the one with better performance 
"""
# PCA With Standardization (X_std)
from sklearn.decomposition import PCA 
pca = PCA(n_components = 3)
pca.fit(X_std)
pca.explained_variance_ratio_
# z1 repressent 18.6% of the existing variation, z2 = 16.9%, z3 = 12.9% 

# PCA Without Standardization (X)
pca2 = PCA(n_components = 3)
pca2.fit(X)
pca2.explained_variance_ratio_
# z1 = 82.1% , z2 = 15.8%, z3 = 2.1%, summation = 100%

###             Task 3 
"""
Split the data into training (70%) and test(30%) with random state = 5
"""
from sklearn.model_selection import train_test_split 
# Note that I will split the standardized predictors as I think the variance without standardization is high
X_train, X_test, y_train, y_test = train_test_split(X_std, y, test_size = 0.3, random_state = 5)



###             Task 4 
"""
Develop K-Nearest Neighbor (K-NN) models where the number of neighbors (k) is 3 using the following:
    Model 1: ALl predictors 
    Model 2: Predictors using LASSO 
    Model 3: Using principal components by PCA 
"""
from sklearn.neighbors import KNeighborsClassifier
# Model 1 All predictors (I will use the Standardized Predictors!)
knn = KNeighborsClassifier( n_neighbors = 3, p = 2, weights='distance') # This means setting the value of K as 3
model1 = knn.fit(X_train, y_train) 

# Model 2  Predictors selected by LASSO 
knn2 = KNeighborsClassifier( n_neighbors = 3, p = 2, weights='distance' ) 
X_LASSO = X_std.loc[:, ['Income', 'CD Account', 'Education']]
X_train2, X_test2, y_train2, y_test2 = train_test_split(X_LASSO, y, test_size = 0.3, random_state = 5)
model2 = knn2.fit(X_train2, y_train2)

# Model 3 PCA (I will develop two different models and select the better performance)
# With Standardization
knn3 = KNeighborsClassifier( n_neighbors = 3, p = 2, weights='distance' ) 
X_PCA_std = pca.transform(X_std) #to get a new dataset of the optimized predictors 
X_train3, X_test3, y_train3, y_test3 = train_test_split(X_PCA_std, y, test_size = 0.3, random_state = 5)
model3 = knn3.fit(X_train3, y_train3)

# Without Standardization 
X_PCA_Unstd = pca2.transform(X) #to get a new dataset of the optimized predictors 
X_train3_Unstd, X_test3_Unstd, y_train3_Unstd, y_test3_Unstd = train_test_split(X_PCA_Unstd, y, test_size = 0.3, random_state = 5)
model3_Unstd = knn3.fit(X_train3_Unstd, y_train3_Unstd)


###             Task 5 
""" 
Report the Performance measures of the 3 Models 
""" 
from sklearn.metrics import accuracy_score
from sklearn import metrics
import timeit

## Model 1 
y_test_pred = model1.predict(X_test)
accuracy_KNN1 = accuracy_score(y_test, y_test_pred)
print(accuracy_KNN1) # 0.958
metrics.precision_score(y_test, y_test_pred) # for precision score 
metrics.recall_score(y_test, y_test_pred) # for recall score 
start = timeit.default_timer()
model1 = knn.fit(X_train, y_train)
stop = timeit.default_timer()
print('Time:',stop - start)


## Model 2  
y_test_pred2 = model2.predict(X_test2)
accuracy_KNN2 = accuracy_score(y_test2, y_test_pred2)
print(accuracy_KNN2) # 0.959
metrics.precision_score(y_test2, y_test_pred2) # for precision score 
metrics.recall_score(y_test2, y_test_pred2)
start = timeit.default_timer()
model2 = knn2.fit(X_train2, y_train2)
stop = timeit.default_timer()
print('Time:',stop - start)

## Model 3 (PCA - Try with and Without Standardization and pick the better accuracy )
# With Standardization: 
y_test_pred3 = model3.predict(X_test3)
accuracy_KNN3 = accuracy_score(y_test3, y_test_pred3)
print(accuracy_KNN3) # 0.901
metrics.precision_score(y_test3, y_test_pred3) # for precision score 
metrics.recall_score(y_test3, y_test_pred3)

# WithOUT Standardizartion: 
y_test_pred3_Unstd = model3_Unstd.predict(X_test3_Unstd)
accuracy_KNN3_Unstd = accuracy_score(y_test3_Unstd, y_test_pred3_Unstd)
print(accuracy_KNN3_Unstd) # 0.897
# ^ Since the auccracy score for Unstandardized model is lower, I will use the Standardized model 

# Task 6 Report the top three most important variables for Model 2 and Model 3 
# Model 2: Income, CD Account, Education 
# Model 3: Unable to be determined since we used PCA   

# Results 

# LASSO resulted in the best performance with the highest accuracy score of 96% and 
# the most influential predictors are Income, CD Account, and Education.



