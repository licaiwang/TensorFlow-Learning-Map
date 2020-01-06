##回歸演示

現在假設有10個x_data和y_data，x和y之間的關係是y_data = b + w *x_data。b，w都是參數，是需要學習出來的。現在我們來練習用梯度下降找到b和w。

```
將numpy導入為np
導入matplotlib.pyplot作為plt
從pylab import mpl

＃matplotlib沒有中文字體，動態解決
plt.rcParams ['font.sans-serif'] = ['Simhei']＃顯示中文
mpl.rcParams ['axes.unicode_minus'] = False＃解決保存圖像是負號'-'顯示為方塊的問題
```

```
x_data = [338.，333.，328.，207.，226.，25.，179.，60.，208.，606.]
y_data = [640.，633.，619.，393.，428.，27.，193.，66.，226.，1591.]
x_d = np.asarray（x_data）
y_d = np.asarray（y_data）
```


```
x = np.arange（-200，-100，1）
y = np.arange（-5、5、0.1）
Z = np.zeros（（len（x），len（y）））
X，Y = np.meshgrid（x，y）
```
  
```
＃ 失利
對於範圍（len（x））中的i：
    對於範圍（len（y））中的j：
        b = x [i]
        w = y [j]
        Z [j] [i] = 0＃meshgrid吐出結果：y為行，x為列
        對於範圍內的n（len（x_data））：
            Z [j] [i] + =（y_data [n]-b-w * x_data [n]）** 2
        Z [j] [i] / = len（x_data）
```



先給b和w一個初始值，計算出b和w的偏微分
python
＃線性回歸
#b = -120
#w = -4
b = -2
w = 0.01
lr = 0.000005
迭代= 1400000

b_history = [b]
w_history = [w]
loss_history = []
導入時間
開始= time.time（）
對於我在範圍內（迭代）：
    m =浮點數（len（x_d））
    y_hat = w * x_d + b
    損失= np.dot（y_d-y_hat，y_d-y_hat）/ m
    grad_b = -2.0 * np.sum（y_d-y_hat）/ m
    grad_w = -2.0 * np.dot（y_d-y_hat，x_d）/ m
    ＃更新參數
    b-= lr * grad_b
    w-= lr * grad_w

    b_history.append（b）
    w_history.append（w）
    loss_history.append（損失）
    如果我％10000 == 0：
        打印（“步驟％i，w：％0.4f，b：％。4f，損失：％。4f”％（i，w，b，損失））
結束= time.time（）
打印（“大約需要時間：”，結束開始）
```
python
＃繪製圖
plt.contourf（x，y，Z，50，alpha = 0.5，cmap = plt.get_cmap（'jet'））＃填充等高線
plt.plot（[-188.4]，[2.67]，'x'，ms = 12，mew = 3，color =“ orange”）
plt.plot（b_history，w_history，'o-'，ms = 3，lw = 1.5，color ='black'）
plt.xlim（-200，-100）
plt.ylim（-5，5）
plt.xlabel（r'$ b $'）
plt.ylabel（r'$ w $'）
plt.title（“線性回歸”）
plt.show（）

```
輸出結果如圖

！[chapter1-0.png]（res / chapter4-1.png）

橫坐標是b，縱坐標是w，標記×位最優解，注意，在圖中我們並沒有運行得到最優解，最優解十分的遙遠。那麼我們就調大學習率，lr = 0.000001（調大10倍），得到結果如下圖。

！[chapter1-0.png]（res / chapter4-2.png）

我們再調大學習率，lr = 0.00001（調大10倍），得到結果如下圖。

！[chapter1-0.png]（res / chapter4-3.png）

結果發現學習率太大了，結果很不好。

所以我們給b和w特製化兩種學習率
python
＃線性回歸
b = -120
w = -4
lr = 1
迭代= 100000

b_history = [b]
w_history = [w]

lr_b = 0
lr_w = 0
導入時間
開始= time.time（）
對於我在範圍內（迭代）：
    b_grad = 0.0
    w_grad = 0.0
    對於範圍內的n（len（x_data））：
        b_grad = b_grad-2.0 *（y_data [n] -n-w * x_data [n]）* 1.0
        w_grad = w_grad-2.0 *（y_data [n] -n-w * x_data [n]）* x_data [n]
    
    lr_b = lr_b + b_grad ** 2
    lr_w = lr_w + w_grad ** 2
    ＃更新參數
    b-= lr / np.sqrt（lr_b）* b_grad
    w-= lr /np.sqrt(lr_w）* w_grad

    b_history.append（b）
    w_history.append（w）
```
python
＃繪製圖
plt.contourf（x，y，Z，50，alpha = 0.5，cmap = plt.get_cmap（'jet'））＃填充等高線
plt.plot（[-188.4]，[2.67]，'x'，ms = 12，mew = 3，color =“ orange”）
plt.plot（b_history，w_history，'o-'，ms = 3，lw = 1.5，color ='black'）
plt.xlim（-200，-100）
plt.ylim（-5，5）
plt.xlabel（r'$ b $'）
plt.ylabel（r'$ w $'）
plt.title（“線性回歸”）
plt.show（）

```

！[chapter1-0.png]（res / chapter4-4.png）

有了新的特製化兩種學習率就可以在10w次迭代之內到達最優點了。
