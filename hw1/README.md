# hw1

###### tags: `ML`

## 傳送門
[作業投影片](https://docs.google.com/presentation/d/18MG1wSTTx8AentGnMfIRUp8ipo8bLpgAj16bJoqW-b0/edit#slide=id.g4cd6560e29_0_26)

[Kaggle 連結](https://www.kaggle.com/c/ml2020spring-hw1)

## 簡介
本次作業使用豐原站的觀測記錄，分成 train set 跟 test set，train set 是豐原站每個月的前 20 天所有資料。test set 則是從豐原站剩下的資料中取樣出來。
train.csv: 每個月前 20 天的完整資料。
test.csv : 從剩下的資料當中取樣出連續的 10 小時為一筆，前九小時的所有觀測數據當作 feature，第十小時的 PM2.5 當作 answer。一共取出 240 筆不重複的 test data，請根據 feature 預測這 240 筆的 PM2.5。


## 作業限制

1. hw1.sh
   numpy 1.17.5 (不可以使用 np.linalg.lstsq)
   
2. hw1_best.sh
   numpy 1.17.5
   torch 1.2.0
   tensorflow 1.15.0
   keras 2.2.5
   scikit-learn 0.22.1

## 結果