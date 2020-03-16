## 上一次失敗的例子
deep learning這麼潮的東西，實現起來也很簡單。先上更新後的 Code
```
import numpy as np
from keras.datasets import mnist
from keras.layers import Conv2D, Flatten, MaxPooling2D
from keras.layers.core import Activation, Dense, Dropout
from keras.models import Sequential
from keras.optimizers import SGD, Adam
from keras.utils import np_utils


def load_data():  # categorical_crossentropy
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    number = 10000
    x_train = x_train[0:number]
    y_train = y_train[0:number]
    x_train = x_train.reshape(number, 28 * 28)
    x_test = x_test.reshape(x_test.shape[0], 28 * 28)
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    # convert class vectors to binary class matrices
    y_train = np_utils.to_categorical(y_train, 10)
    y_test = np_utils.to_categorical(y_test, 10)
    x_train = x_train
    x_test = x_test
    
    # 255是灰階的表示，除以 255 才能 normalize 到 0-1 之間
    x_train = x_train / 255
    x_test = x_test / 255

    # 加上 noise 擾亂原本的輸出，沒加 dropout 的話 Test ACC會掉到約 40%
    x_test = np.random.normal(x_test)
    
    return (x_train, y_train), (x_test, y_test)
    
def main():
    (x_train, y_train), (x_test, y_test) = load_data()

    # define network structure
    model = Sequential()
    # add dropout
    model.add(Dropout(0.5))
    # activation ='relu' , 'sigmoid'
    model.add(Dense(input_dim=28 * 28, units=600, activation='relu'))
    # model.add(Dropout(0.5))
    model.add(Dense(units=600, activation='relu'))
    # model.add(Dropout(0.5))
    model.add(Dense(units=10, activation='softmax'))

    # set configurations
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam', metrics=['accuracy'])
    # loss = 'categorical_crossentropy','mse'
    # train model
    model.fit(x_train, y_train, batch_size=100, epochs=20)

    # evaluate the model and output the accuracy
    result_train = model.evaluate(x_train, y_train)
    result_test = model.evaluate(x_test, y_test)
    print('Train Acc:', result_train[1])
    print('Test Acc:', result_test[1])
main()
```
![在這裡插入圖片描述](./res/chapter19_1.png)

結果是差的，那麼該怎麼辦。首先先看你在train data的performer，如果它在train data上做得好，那麼可能是過擬合，如果在train data上做得不好，怎麼能讓它做到舉一反三呢。所以我們至少先讓它在train data 上得到好的結果。
```
model.evaluate(x_train,y_train,batch_size=10000)
```
![在這裡插入圖片描述](./res/chapter19_2.png)

train data acc 也是差的，就說明train沒有train好，並不是overfiting
## 調參過程
### loss function
```
model.compile(loss='categorical_crossentropy',optimizer=SGD(lr=0.1),metrics=['accuracy'])
```
分類問題mse不適合，將loss mse function 改為categorical_crossentropy，看看有怎樣的差別

![在這裡插入圖片描述](./res/chapter19_3.png)

當我們一換categorical_crossentropy，在train set上的結果就起飛了。得到87.34%的正確率，現在就比較有train起來了。
### batch_size
再試一下batch_size對結果的影響，現在我們的batch_size是100，改成10000試試看

```
model.fit(x_train,y_train,batch_size=10000,epochs=20)
```
![在這裡插入圖片描述](./res/chapter19_4.png)

batch_size 設10000，跑超快，然而一樣的架構，batch_size太大的時候performer就壞掉。再把10000改為1
```
model.fit(x_train,y_train,batch_size=1,epochs=20)
```
GPU沒有辦法利用它的並行運算，所以跑得超慢~
### deep layer
再看看deep layer，我們再加10層
```
for _ in range(10):
model.add(Dense(units=689,activation='sigmoid'))

```
![在這裡插入圖片描述](./res/chapter19_5.png)

沒有train 起來~~接著改下activation function
### activation function
我們把sigmoid都改為relu，發現現在train的accuracy就爬起來了，train的acc已經將近100分了，test 上也可以得到95.64%
![在這裡插入圖片描述](./res/chapter19_6.png)

### normalize
現在的圖片是有進行normalize，每個pixel我們用一個0-1之間的值進行表示，那麼我們不進行normalize，把255拿掉會怎樣呢？
```
# x_train=x_train/255
# x_test=x_test/255
```
![在這裡插入圖片描述](./res/chapter19_7.png)

你會發現你又做不起來了，所以這種小小的地方，只是有沒有做normalizion，其實對你的結果會有關鍵性影響。

### optimizer
把SGD改為Adam，然後再跑一次，你會發現說，用adam的時候最後收斂的地方查不到，但是上升的速度變快。

![在這裡插入圖片描述](./res/chapter19_8.png)

### Random noise
在test set上每個pixel上隨機加noise，再看看結果會掉多少
```
x_test=np.random.normal(x_test)
```

![在這裡插入圖片描述](./res/chapter19_9.png)

結果就爛掉了，over fiting 了~
### dropout
我們再試試dropout能帶來什麼效果
```
model.add(Dense(input_dim=28*28,units=689,activation='relu'))
model.add(Dropout(0.7))
model.add(Dense(units=689,activation='relu'))
model.add(Dropout(0.7))
model.add(Dense(units=689,activation='relu'))
model.add(Dropout(0.7))
model.add(Dense(units=10,activation='softmax'))

```

dropout 加在每個hidden layer，要知道dropout加入之後，train的效果會變差，然而test的正確率提升了

![在這裡插入圖片描述](./res/chapter19_10.png)

不同的tip對效果有不同的影響，應該要多試試
