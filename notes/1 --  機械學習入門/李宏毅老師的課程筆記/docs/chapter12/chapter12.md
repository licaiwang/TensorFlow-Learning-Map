# 贏家還是輸家：

本次作業是需要從給定的個人資訊（如下表），預測此人的年收入是否大於50K。

共有32561筆訓練資料，16281筆測試資料

## 數據集和任務描述

- [Data Link](https://drive.google.com/file/d/0B8Si647wj9ZoTE9uQzAwR0M5ZkU/view?usp=sharing)

- 二分類問題，判斷一個人年薪是否超過 50k

由Barry Becker從1994年人口普查數據庫中提取，按照((AGE>16) && (AGI>100) && (AFNLWGT>1) && (HRSWK>0))條件過濾後，保持數據的干淨。

[來源](https://archive.ics.uci.edu/ml/datasets/Adult)

## 特徵屬性描述
數據一共包含13個特徵，一個年薪是否超過50K的label
train.csv 、test.csv :
age, workclass, fnlwgt, education, education num, marital-status, occupation
relationship, race, sex, capital-gain, capital-loss, hours-per-week,
native-country, make over 50K a year or not
![12-1](./res/chapter12-1.png)

## 抽取後的特徵
- 離散數據進行one-hot編碼，如work_class,education...
- 連續特徵保持不變，如age,capital_gain...
- X_train,X_test 每個樣本包含106維特徵，一個樣本作為一行
- Y_train:label=0 表示年薪低於等於50k,label=1 表示年薪高於50K
![12-2](./res/chapter12-2.png)


## [參考代碼](https://github.com/orbxball/ML2017/tree/master/hw2)

