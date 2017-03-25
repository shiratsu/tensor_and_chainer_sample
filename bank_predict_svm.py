# coding: UTF-8

import numpy as np
import pandas as pd
from sklearn.cross_validation import train_test_split
import skutil
from sklearn import svm
from sklearn import metrics

# データの読み込み
bank = pd.read_csv("data/bank-full.csv",sep=";")
# print(bank.head(3))

features,label = skutil.makeFeatures(bank.drop('y',1)),bank.y
# print(features)
# print(label)

random_state = np.random.RandomState(123)
X_train,X_test,y_train,y_test = train_test_split(features,label,test_size=.3,random_state=random_state)

# RBFカーネルによる予測モデルの構築
clf = svm.SVC()
clf.fit(X_train,y_train)

# 予測
pred = clf.predict(X_test)

print(metrics.classification_report(y_test,pred,target_names=['no','yes']))
