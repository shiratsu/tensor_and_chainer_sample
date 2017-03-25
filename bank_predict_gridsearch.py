# coding: UTF-8

import numpy as np
import pandas as pd
from sklearn.cross_validation import train_test_split
import skutil
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn import grid_search as gs

# データの読み込み
bank = pd.read_csv("data/bank-full.csv",sep=";")
# print(bank.head(3))

features,label = skutil.makeFeatures(bank.drop('y',1)),bank.y
# print(features)
# print(label)

random_state = np.random.RandomState(123)
X_train,X_test,y_train,y_test = train_test_split(features,label,test_size=.3,random_state=random_state)

parameters = {
        'n_estimators'      : [5, 10, 20, 30, 50, 100, 300],
        'max_features'      : [3, 5, 10, 15, 20],
        'random_state'      : [random_state],
        'n_jobs'            : [1],
        'min_samples_split' : [3, 5, 10, 15, 20, 25, 30, 40, 50, 100],
        'max_depth'         : [3, 5, 10, 15, 20, 25, 30, 40, 50, 100]
}

# RandomForestによるyosoku
clf = gs.GridSearchCV(RandomForestClassifier(), parameters)
clf.fit(X_train,y_train)

# 予測
pred = clf.predict(X_test)

print(metrics.classification_report(y_test,pred,target_names=['no','yes']))
print(clf.best_estimator_)
