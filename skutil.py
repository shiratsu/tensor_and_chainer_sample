# coding: UTF-8

import numpy as np
import pandas as pd

def makeFeatures(x):

    # 数値変数のスケーリング
    cn_num = ['age','balance','day','duration','campaign','pdays','previous']
    x_num = x[cn_num]

    x[cn_num] = (x_num-x_num.mean())/x_num.std()

    # ダミー変数の変換
    x_dum = pd.get_dummies(x)
    return x_dum
