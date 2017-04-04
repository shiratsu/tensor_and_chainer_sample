# 数値計算関連
import math
import random
import numpy as np
import matplotlib.pyplot as plt
# chainer
from chainer import Chain, Variable
import chainer.functions as F
import chainer.links as L
from chainer import optimizers

# -*- coding: utf-8 -*-
class MyChain(Chain):

    def __init__(self):

        # この場合は層が４つで出力層（４つ目が出力層）
        super(MyChain, self).__init__(
            l1 = L.Linear(1, 100),
            l2 = L.Linear(100, 30),
            l3 = L.Linear(30, 1)
        )

    def predict(self, x):
        z1 = F.relu(self.l1(x))
        h2 = F.relu(self.l2(z1))
        return self.l3(h2)

# 乱数のシードを固定
random.seed(1)

# 標本データの生成
#   真の関数としてsin関数を
x, y = [], []

# np.linspace(-3,3,100):-3から3までの100のデータを作っている
for i in np.linspace(-3,3,100):
    x.append([i])
    y.append([math.sin(i)])  # 真の関数
# chainerの変数として再度宣言
x = Variable(np.array(x, dtype=np.float32))
y = Variable(np.array(y, dtype=np.float32))


# NNモデルを宣言
model = MyChain()


# 損失関数の計算
# 損失関数には自乗誤差(MSE)を使用
# 評価関数 = 損失関数
def forward(x, y, model):
    # print(x.data)
    # 予測値を計算
    t = model.predict(x)
    # 誤差を出す
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
for i in range(0,1000):
    loss = forward(x, y, model) # 順伝播
    # print(loss.data)  # 現状のMSEを表示
    optimizer.update(forward, x, y, model) # 逆伝播

# # プロット
t = model.predict(x)
print(t.data)
# # plotの方がグラフ
# plt.plot(x.data, y.data) # 実測値
#
# # scatterの方が散布図
# plt.scatter(x.data, t.data) # 予測値
# plt.grid(which='major',color='gray',linestyle='-')
# plt.ylim(-1.5, 1.5)
# plt.xlim(-4, 4)
# plt.show()

# 未来のデータはｘ（入力変数）のデータだけ用意して、predictを叩けば良い
