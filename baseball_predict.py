# coding: utf-8
import numpy as np
import random

import pandas as pd
import chainer
from chainer import Variable,training
from chainer import optimizers
from chainer import Chain
import chainer.functions as F
import chainer.links as L
from chainer.training import extensions

class BaseBallChain(Chain):

    def __init__(self):

        # この場合は層が４つで出力層（４つ目が出力層）
        super(BaseBallChain, self).__init__(
            l1 = L.Linear(33, 40),
            l2= L.Linear(40, 5),
            l3= L.Linear(5, 1),
        )
    # def __call__(self, x):
    #     z1 = F.relu(self.l1(x))
    #     return self.l2(z1)

    def predict(self, x):
        # print("--------------------------xのデータ----------------------------------------")
        # print(x.data)
        h1 = F.relu(self.l1(x))
        h2 = F.relu(self.l2(h1))
        y = self.l3(h2)
        return y

# 乱数のシードを固定
random.seed(1)
np.random.seed(1)


TRAIN_DATA_SIZE = 90

raw_input = np.loadtxt(open("data/input.csv"), delimiter=",")
[salary, score]  = np.hsplit(raw_input, [1])

[salary_train, salary_test] = np.vsplit(salary, [TRAIN_DATA_SIZE])
[score_train, score_test] = np.vsplit(score, [TRAIN_DATA_SIZE])

# float32にキャストしないとtype errorになる
salary_train = salary_train.astype(np.float32)
salary_test = salary_test.astype(np.float32)
score_train = score_train.astype(np.float32)
score_test = score_test.astype(np.float32)

# NNモデルを宣言
model = BaseBallChain()



# 損失関数の計算
# 損失関数には自乗誤差(MSE)を使用
# 評価関数 = 損失関数
def forward(x, y, model):
    # print(x.data)
    # 予測値を計算
    t = model.predict(x)
    loss = F.mean_squared_error(t, y)
    return loss

# chainerのoptimizer
#   最適化のアルゴリズムには Adam を使用
# 例えば、最急降下法なら
# optimizer = optimizers.SGD() #多分合ってる
# Adamが一番良い
optimizer = optimizers.Adam()
# modelのパラメータをoptimizerに渡す
optimizer.setup(model)


# パラメータの学習を繰り返す
# 1000回繰り返す
for i in range(0,450):
    x = Variable(score_train)
    y = Variable(salary_train)
    loss = forward(x, y, model) # 順伝播
    print(loss.data)  # 現状のMSEを表示
    optimizer.update(forward, x, y, model) # 逆伝播

# あとはテストパラメータで試してみる
# 誤差が小さければ、多分正しいはず
xtest = Variable(score_test)
ytest = Variable(salary_test)
ypredict = model.predict(xtest)
print(ypredict.data)
print(salary_test)

loss = F.mean_squared_error(ypredict, ytest)
print(loss.data)
