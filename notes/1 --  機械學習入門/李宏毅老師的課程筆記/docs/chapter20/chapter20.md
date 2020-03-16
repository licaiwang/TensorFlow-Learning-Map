## Fizz Buzz Google 面試題
對數字101到1000做了labeling，即訓練數據xtrain.shape=(900,10)，每一個數字都是用二進位來表示，第一個數字是101，用二進位來表示即為[1, 0,1,0,0,1,1,0,0,0]，每一位表示2^(n-1)，n 表示左數第幾位。現在一共有四個case，[一般，Fizz，Buzz，Fizz Buzz]，所以y_train.shape=(900,10)，對應的維度用1表示，其他都為0

## Code
```
from keras.layers.normalization import BatchNormalization
from keras.models import Sequential
from keras.layers.core import Dense,Dropout,Activation
from keras.optimizers import SGD,Adam
import numpy as np

def fizzbuzz(start,end):
    x_train,y_train=[],[]
    for i in range(start,end+1):
        num = i
        tmp=[0]*10
        j=0
        while num :
            tmp[j] = num & 1
            num = num>>1
            j+=1
            x_train.append(tmp)
            if i % 3 == 0 and i % 5 ==0:
                y_train.append([0,0,0,1])
            elif i % 3 == 0:
                y_train.append([0,1,0,0])
            elif i % 5 == 0:
                y_train.append([0,0,1,0])
            else :
                y_train.append([1,0,0,0])
    return np.array(x_train),np.array(y_train)

def main():
    x_train,y_train = fizzbuzz(101,1000) #打標記函數
    x_test,y_test = fizzbuzz(1,100)
    model = Sequential()
    model.add(Dense(input_dim=10,output_dim=1000))
    model.add(Activation('relu'))
    model.add(Dense(output_dim=4))
    model.add(Activation('softmax'))
    model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
    model.fit(x_train,y_train,batch_size=20,nb_epoch=100)
    result = model.evaluate(x_test,y_test,batch_size=1000)
    print('Acc：',result[1])
main()
```

![在這裡插入圖片描述](res/chapter20_1.png)

結果並沒有達到百分百正確率，然而並不會放棄，所以我們首先開一個更大的neure，把hidden neure 從100改到1000

```
model.add(Dense(input_dim=10,output_dim=1000))
```

再跑一跑，跑起來了，跑到100了，正確率就是100分

![在這裡插入圖片描述](res/chapter20_2.png)
