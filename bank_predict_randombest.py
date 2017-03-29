# coding: UTF-8

import numpy as np
import pandas as pd
from sklearn.cross_validation import train_test_split
import skutil
from sklearn import ensemble
from sklearn import metrics

# データの読み込み
bank = pd.read_csv("data/bank-full.csv",sep=";")
# print(bank.head(3))

features,label = skutil.makeFeatures(bank.drop('y',1)),bank.y
# print(features)
# print(label)

random_state = np.random.RandomState(123)
X_train,X_test,y_train,y_test = train_test_split(features,label,test_size=.3,random_state=random_state)

# RandomForestによるyosoku
clf = ensemble.RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
            max_depth=40, max_features=15, max_leaf_nodes=None,
            min_impurity_split=1e-07, min_samples_leaf=1,
            min_samples_split=10, min_weight_fraction_leaf=0.0,
            n_estimators=100, n_jobs=1, oob_score=False,
            random_state=random_state,
            verbose=0, warm_start=False)
clf.fit(X_train,y_train)

# 予測
pred = clf.predict(X_test)

print(metrics.classification_report(y_test,pred,target_names=['no','yes']))
