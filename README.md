# 2021-DSAI-HW2作業說明

## 使用說明
安裝相依套件
```
pip install -r requirements.txt
```
運行主程式
```
python trader.py --training training_data.csv -- testing testing_data.csv --output output.csv
```

## 模型說明

### LSTM

#### 模型介紹
長短期記憶模型，主要用於時間序列資料的預測，可以把連續的資料集切隔成一個個連續的窗格來進行趨勢預測，本專案將窗格大小設定為3，這意味著可運用前三日的資料來預測隔日的資料。

#### 資料準備
使用IBM股票的歷史資料來當訓練集，資料欄位依序分別為「開盤價」、「當日最高價」、「當日最低價」、「收盤價」。而我們使用「開盤價」最為模型所需的「ytrain欄位」，而把每日開盤價切割成一個一個窗格作為模型所需的「xtrain欄位」，再丟入模型進行訓練。

#### 參數設定
Batch_size設定為 30
Epochs設定為100
Kernel_initializer使用glorot_uniform


#### 訓練結果
我們從IBM股票的歷史資料中，最後的20筆資料作為我們的驗證集，最終模型預測出來的結果經RMSE的計算後可得到1.21，圖表則如下所示：
![](https://i.imgur.com/XytD4aC.png)