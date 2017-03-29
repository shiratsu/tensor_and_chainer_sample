# tensor_and_chainer_sample
tensorflowとchainerで遊ぶリポジトリ
以下はタイタニックの生存予測のやつ

## chainerの結果
```
epoch       main/accuracy  validation/main/accuracy
1           0.605714       0.688824
2           0.673333       0.650588
3           0.673333       0.690588
4           0.65           0.698824
5           0.688571       0.690588
6           0.701667       0.70549
7           0.68           0.687255
8           0.675          0.712353
9           0.698571       0.720392
10          0.688333       0.718824
11          0.683333       0.707255
12          0.685          0.722157
13          0.691667       0.702157
14          0.68           0.702353
15          0.703333       0.718627
16          0.688333       0.715294
17          0.691667       0.715294
18          0.664286       0.711961
19          0.703333       0.721961
20          0.71           0.721961
21          0.676667       0.69549
22          0.7            0.71549
23          0.691667       0.707059
24          0.691667       0.662353
25          0.698333       0.756667
26          0.695          0.693725
27          0.667143       0.702157
28          0.703333       0.702353
29          0.693333       0.708824
30          0.7            0.705294
```

### chainerでの入力変数の渡し方
```
x = Variable(np.array(x, dtype=np.float32))
```

## tensorでの結果
```
I tensorflow/core/common_runtime/local_device.cc:40] Local device intra op parallelism threads: 8
I tensorflow/core/common_runtime/direct_session.cc:58] Direct session inter op parallelism threads: 8
Epoch: 0001 cost= 12.056934711
Epoch: 0002 cost= 3.682290173
Epoch: 0003 cost= 2.288028301
Epoch: 0004 cost= 1.445026050
Epoch: 0005 cost= 1.118387491
Epoch: 0006 cost= 0.955024500
Epoch: 0007 cost= 1.387049086
Epoch: 0008 cost= 0.833590559
Epoch: 0009 cost= 0.490418169
Epoch: 0010 cost= 0.371005500
Optimization Finished!
Accuracy: 0.736264
```

## SVMの結果の評価

### 正解率(Accuracy)
予測のうち、正しく分類されたものの割合

### 適合率(Precision)
正と予測したデータのうち，実際に正であるものの割合

### 再現率(Recall)
実際に正であるもののうち，正であると予測されたものの割合

### F値(F-measure)
精度と再現率の調和平均

## grid search
分類モデルの最適なパラメータを見つける
http://blog.tatsushim.com/?p=63

# grid searchの結果
結果の意味はあとで確認
```
~/chainer_sample on  master! ⌚ 13:16:34
$ python3 bank_predict_gridsearch.py                                                                                                                                                     2.3.1
/usr/local/lib/python3.6/site-packages/sklearn/cross_validation.py:44: DeprecationWarning: This module was deprecated in version 0.18 in favor of the model_selection module into which all the refactored classes and functions are moved. Also note that the interface of the new CV iterators are different from that of this module. This module will be removed in 0.20.
  "This module will be removed in 0.20.", DeprecationWarning)
/usr/local/lib/python3.6/site-packages/sklearn/grid_search.py:43: DeprecationWarning: This module was deprecated in version 0.18 in favor of the model_selection module into which all the refactored classes and functions are moved. This module will be removed in 0.20.
  DeprecationWarning)
             precision    recall  f1-score   support

         no       0.93      0.97      0.95     11998
        yes       0.63      0.44      0.52      1566

avg / total       0.90      0.91      0.90     13564

RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
            max_depth=40, max_features=15, max_leaf_nodes=None,
            min_impurity_split=1e-07, min_samples_leaf=1,
            min_samples_split=10, min_weight_fraction_leaf=0.0,
            n_estimators=100, n_jobs=1, oob_score=False,
            random_state=<mtrand.RandomState object at 0x111b65360>,
            verbose=0, warm_start=False)
```
