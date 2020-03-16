## 手寫數字辨識
deep learning這麼潮的東西，實現起來也很簡單。
```
import numpy as np
from keras.models import Sequential
from keras.layers.core import Dense,Dropout,Activation
from keras.optimizers import SGD,Adam
from keras.utils import np_utils
from keras.datasets import mnist

def load_data():
	(x_train,y_train),(x_test,y_test)=mnist.load_data()
	number=10000
	x_train=x_train[0:number]
	y_train=y_train[0:number]
	x_train=x_train.reshape(number,28*28)
	x_test=x_test.reshape(x_test.shape[0],28*28)
	x_train=x_train.astype('float32')
	x_test=x_test.astype('float32')
	y_train=np_utils.to_categorical(y_train,10)
	y_test=np_utils.to_categorical(y_test,10)
	x_train=x_train
	x_test=x_test
	x_train=x_train/255
	x_test=x_test/255
	return (x_train,y_train),(x_test,y_test)

	(x_train,y_train),(x_test,y_test)=load_data()
def main():
	model=Sequential()
	model.add(Dense(input_dim=28*28,units=633,activation='sigmoid'))
	model.add(Dense(units=633,activation='sigmoid'))
	model.add(Dense(units=633,activation='sigmoid'))
	model.add(Dense(units=10,activation='softmax'))

	model.compile(loss='mse',optimizer=SGD(lr=0.1),metrics=['accuracy'])

	model.fit(x_train,y_train,batch_size=100,epochs=20)

	result= model.evaluate(x_test,y_test)

	print('TEST ACC:',result[1])
main()
```

其中x_train是一個二維的向量，x_train.shape=(10000,784)，這個是什麼意思呢，就告訴我們現在train data一共有一萬筆，每筆由一個784維的vector所表示。 y_train也是一個二維向量，y_train.shape=(10000,10)，其中只有一維的數字是1，其餘的為0。結果如下圖
![在這裡插入圖片描述](./res/chapter17_1 .png)
正確率只有11.35%，感覺不太行，這個時候就開始焦躁了，調一下參數~~~
## 調參過程
### 隱層神經元個數
```
model.add(Dense(input_dim=28*28,units=689,activation='sigmoid'))
model.add(Dense(units=689,activation='sigmoid'))
model.add(Dense(units=689,activation='sigmoid'))
model.add(Dense(units=10,activation='softmax'))
```
![在這裡插入圖片描述](./res/chapter17_2.png)

結果如上，似乎好一點了，那好一點就繼續~
### 深度
deep learning 就是很deep的樣子，那麼才三層，用for添加10層
```
model.add(Dense(input_dim=28*28,units=689,activation='sigmoid'))
model.add(Dense(units=689,activation='sigmoid'))
model.add(Dense(units=689,activation='sigmoid'))
for _ in range(10):
model.add(Dense(units=689,activation='sigmoid'))

model.add(Dense(units=10,activation='softmax'))
```
![在這裡插入圖片描述](./res/chapter17_3.png)

哎，結果還是10%左右這樣子，然後你就開始焦躁不安。參數調來調去，發現什麼東西都沒有做出來，最後從入門到放棄這樣。

## 總結
- deep learning 並不是越deep越好
- 隱層Neure調整，對整體效果也不一定有助益
- 關於deep learning 的實踐，還是需要基於理論基礎，而不是參數隨便調來調去，所以繼續跟著課程好好學。
