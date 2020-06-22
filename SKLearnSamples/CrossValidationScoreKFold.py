# -*- coding: utf-8 -*-
"""
Created on Mon Jun  1 13:33:52 2020

@author: rahul
"""


import numpy as np
from sklearn.model_selection import cross_val_score, KFold
from sklearn import datasets,svm

X, y = datasets.load_digits(return_X_y=True)

svc = svm.SVC(kernel='linear')
C_s = np.logspace(-10, 0, 10)

k_fold = KFold(n_splits=5)

cv_score = cross_val_score(svc,X,y,cv=k_fold,n_jobs=-1)
print(cv_score)