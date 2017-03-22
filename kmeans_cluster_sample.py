# coding: UTF-8
import operator
import numpy as np
from sklearn.cluster import KMeans

# 生徒の国語・数学・英語の各得点を配列として与える
features = np.array([
        [  80,  85, 100 ],
        [  96, 100, 100 ],
        [  54,  83,  98 ],
        [  80,  98,  98 ],
        [  90,  92,  91 ],
        [  84,  78,  82 ],
        [  79, 100,  96 ],
        [  88,  92,  92 ],
        [  98,  73,  72 ],
        [  75,  84,  85 ],
        [  92, 100,  96 ],
        [  96,  92,  90 ],
        [  99,  76,  91 ],
        [  75,  82,  88 ],
        [  90,  94,  94 ],
        [  54,  84,  87 ],
        [  92,  89,  62 ],
        [  88,  94,  97 ],
        [  42,  99,  80 ],
        [  70,  98,  70 ],
        [  94,  78,  83 ],
        [  52,  73,  87 ],
        [  94,  88,  72 ],
        [  70,  73,  80 ],
        [  95,  84,  90 ],
        [  95,  88,  84 ],
        [  75,  97,  89 ],
        [  49,  81,  86 ],
        [  83,  72,  80 ],
        [  75,  73,  88 ],
        [  79,  82,  76 ],
        [ 100,  77,  89 ],
        [  88,  63,  79 ],
        [ 100,  50,  86 ],
        [  55,  96,  84 ],
        [  92,  74,  77 ],
        [  97,  50,  73 ],
        ])

# K-means クラスタリングをおこなう
# この例では 3 つのグループに分割 (メルセンヌツイスターの乱数の種を 10 とする)
kmeans_model = KMeans(n_clusters=3, random_state=10).fit(features)

# 分類先となったラベルを取得する
labels = kmeans_model.labels_

# ラベル (班) 、成績、三科目の合計得点を表示する
# for label, feature in zip(labels, features):
#     print(label, feature, feature.sum())

# ソートして出力
newlist = zip(labels, features)
sortedList = sorted(newlist, key=lambda pair: pair[0])
for label, feature in sortedList:
    print(label, feature, feature.sum(),feature.mean())
