## 作業描述
> 讓機器預測到豐原站在下一個小時會觀測到的PM2.5。舉例來說，現在是2017-09-29 08：00：00 ，那麼要預測2017-09-29 09：00：00豐原站的PM2.5值會是多少。

## 作業要求

- 任務要求：**預測PM2.5的值**，我們將用**梯度下降法** (**Gradient Descent**) **預測PM2.5** 的值(**Regression** 回歸問題)
- 環境要求：
  - 要求 **python3.5+**
  - 只能用
    - numpy
    - scipy
    - pandas
  - 請用梯度下降**手寫線性回歸**
  - 最好使用 **Public Simple Baseline**
  - 對於想加載模型而並不想運行整個訓練過程的人：
    - 請上傳訓練代碼並命名成 `train.py`
    - 只要用梯度下降的代碼就行了
- 最佳要求：
  - 要求 **python3.5+**
  - 任何庫都可以用
  - 在 **Kaggle** 上獲得你選擇的更高的分
- 數據介紹：
  本次作業使用豐原站的觀測記錄，分成**train set** 跟**test set**，train set 是豐原站每個月的前20天所有資料，test set則是從豐原站剩下的資料中取樣出來。
  **train.csv**:每個月前20天每個小時的氣象資料(每小時有18種測資)。共12個月。
  **test**.csv:從剩下的資料當中取樣出連續的10小時為一筆，前九小時的所有觀測數據當作feature，第十小時的PM2.5當作answer。一共取出240筆不重複的 test data，請根據feauure預測這240筆的PM2.5。
- 請完成之後參考以下資料：
  - [Sample_code](https://github.com/datawhalechina/leeml-notes/tree/master/docs/Homework/HW_1)


- 特徵選擇

  選擇最具代表性的特徵：**PM10**、**PM2.5**、**SO2**

- 模型建立

  建立線性回歸模型
  
  $$
  y=b+\sum_{i=1}^{27} \mathcal{W}_{i} \times \mathcal{X}_{i} \\
  $$
  
  等價於
  
  $$
  \mathrm{y}=b+w_{1} \times x_{1}+w_{2} \times x_{2}+\cdots+w_{27} \times x_{27}
  $$
  
  其中x1到x9是前九個時間點的PM10值，x10到x18是前9個時間點的PM2.5值，x19到x27是前9個時間點的SO2值，w為對應參數，b為偏移量

- 定義損失函數 (Gradient Descent)

  $$
  L \mathrm{oss}=\frac{1}{2} \sum_{i=1}^{m}\left(y_{i}-y_{\text {ireal}}\right)^{2}
  $$
  
  其中m為每次更新參數時使用的樣本數,yi為預測值，yireal為真實值

  採用小批量梯度下降算法，並且設定批量樣本大小為50，即每次隨機在訓練樣本中選取50個用來更新參數

  設定學習率learning rate分別為0.000000001、0.0000001、0.000001時，比較不同的learning rate對損失函數收斂速度的影響

- 模型評估

  $$
  \mathrm{Model_-Evaluation}=\frac{1}{n} \sum_{i=1}^{n}\left(y_{i}-y_{\text {ireal}}\right)^{2}
  $$

## 總結

- 小批量梯度下降算法中，初始參數的選擇很重要，不同的初始參數，其對應損失函數收斂速度也不一樣
- learning rate 採用遞減的方式選取的，根據經驗的選擇也很重要，說起來挺輕鬆的一件事，但實際操作起來，卻四處碰壁，希望大家在實驗中，多積累、多總結，機器學習不就是坑起坑落，挖坑、填坑、再挖坑、再填坑的反複訓練過程麼~v_v~
