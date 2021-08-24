# -*- coding: utf-8 -*-
"""
Created on Tue Aug 24 14:32:03 2021

@author: hp
"""


#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#%%
from sklearn.datasets import make_classification
X, y = make_classification(n_samples=1000, n_features=5)
print(X)
print(y)

#%%
# split X and y into training and testing sets
from sklearn.model_selection import train_test_split 
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=0)

#%%
# import the class
from sklearn.linear_model import LogisticRegression

# instantiate the model 
logreg = LogisticRegression()
#%%
# fit the model with data
logreg.fit(X_train,y_train)

#%%
y_pred=logreg.predict(X_test)

#%%
# import the metrics class
from sklearn import metrics
cnf_matrix = metrics.confusion_matrix(y_test, y_pred)
print(cnf_matrix)

#%%
print(logreg.intercept_)
print(logreg.coef_)
#%%
from sklearn.metrics import accuracy_score
print(accuracy_score(y_test,y_pred))

#%%
#Maths behind logistic regression 
b1=np.sum(0.5*X_train+0.8,axis=1)
deno=1+np.exp(-b1)
b2=1/deno
print(b2)

#%%
#cost computation
cost=np.sum((y_train)*np.log(b2) + (1- y_train)*np.log(1-b2))/(y_train.shape[0])
print("cost is", cost)








