#!/usr/bin/env python
# coding: utf-8

# In[8]:


import numpy as np
from numpy import genfromtxt
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_curve, auc, roc_auc_score
from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV

scaler = StandardScaler()
#Loading in data
train_data = pd.read_csv('fashion-mnist_train.csv')
test_data = pd.read_csv('fashion-mnist_test.csv')

#Setting up features and scaling them
X_train = np.array(train_data.iloc[:, 1:])
X_test = np.array(test_data.iloc[:, 1:])
X_train = scaler.fit_transform(X_train)
X_test = scaler.fit_transform(X_test)
y_train = np.array(train_data['label'])
y_test = np.array(test_data['label'])

#Converting labels to be of shape m x 10 instead of having label values as a single integer between 0-9
y_train = y_train.reshape(-1)
y_train = np.eye(10)[y_train]
y_test = y_test.reshape(-1)
y_test = np.eye(10)[y_test]

#Cross validation data setup
cross_val_input = X_train[40000:, :]
cross_val_labels = y_train[40000:, :]
X_train = X_train[:40000, :]
y_train = y_train[:40000, :]

#Settings for GridSearch
"""
parameters = {'hidden_layer_sizes': [(50, 50), (350)], 'activation': ['relu'], 'alpha': [0.0001, 0.001, 0.003, 0.01, 0.03], 
              'learning_rate_init': [0.001, 0.003, 0.01, 0.03], 'max_iter': [200], 'n_iter_no_change': [5], 'early_stopping': [True]
              'solver': ['adam'], 'batch_size': [32, 64, 128], 'random_state': [1], 'verbose': [5]}
clf = GridSearchCV(MLPClassifier(), parameters, n_jobs=-1, scoring='f1_weighted')
"""

#model = MLPClassifier(hidden_layer_sizes=(50, 50), batch_size=64, max_iter=200, solver='adam', 
                      #alpha=0.01, learning_rate_init=0.003, verbose=True, random_state=1)

#Best settings from grid search
grid_searched_model = MLPClassifier(hidden_layer_sizes=(350), batch_size=128, max_iter=200, solver='adam', 
                      alpha=0.001, learning_rate_init=0.001, verbose=True, n_iter_no_change=5, random_state=1)
grid_searched_model.fit(X_train, y_train) # training targets

#print (clf.best_params_)
print ("")
print ("---------------CROSS-VALIDATION SCORES-------------")
print (classification_report(cross_val_labels, grid_searched_model.predict(cross_val_input)))
print (accuracy_score(cross_val_labels, grid_searched_model.predict(cross_val_input)))
print (roc_auc_score(cross_val_labels, grid_searched_model.predict(cross_val_input)))
print ("")
print ("---------------TEST-SET SCORES-------------")
predict = grid_searched_model.predict(X_test)
print (accuracy_score(y_test, predict))
print (roc_auc_score(y_test, predict))
print (classification_report(y_test,predict))


# In[ ]:




