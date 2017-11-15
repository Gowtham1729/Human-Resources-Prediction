# -*- coding: utf-8 -*-
"""
Created on Tue Oct 17 19:52:06 2017

@author: Gowtham
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv("HR_comma_sep.csv")

X = dataset.iloc[:,[0,1,2,3,4,5,7,8,9] ].values
y = dataset.iloc[:, 6].values

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X  = LabelEncoder()
X[:, 7] = labelencoder_X.fit_transform(X[:, 7])
X[:, 8] = labelencoder_X.fit_transform(X[:, 8])
onehotencoder = OneHotEncoder(categorical_features = [7, 8])
X = onehotencoder.fit_transform(X).toarray()

from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

from sklearn.svm import SVC
clf = SVC(kernel = 'rbf', random_state = 0)
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)

from sklearn.metrics import accuracy_score
accuracy_svm = accuracy_score(y_test, y_pred, normalize = True)

from sklearn.naive_bayes import GaussianNB
clf1 = GaussianNB()
clf1.fit(X_train, y_train)
y_pred_nb = clf.predict(X_test)

accuracy_nb = accuracy_score(y_test, y_pred_nb, normalize = True)

from sklearn.neighbors import KNeighborsClassifier
clf2 = KNeighborsClassifier(n_neighbors = 6)
clf2.fit(X_train, y_train)

y_pred_knn = clf2.predict(X_test)
accuracy_knn = accuracy_score(y_test, y_pred_knn, normalize = True)
print(accuracy_knn)

from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators = 11, criterion = 'entropy', random_state = 0)
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred_rf = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
accuracy_rf = accuracy_score(y_test, y_pred_rf, normalize = True)