#keras13_lstm.py
import numpy as np
import pandas as pd
import tensorflow as tf
import keras
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout

a = np.array(range(11,21))

window_size = 5                                     # 데이터 5개씩 자르기용     # 입력이 5이고 5개씩 자르기
def split_5(seq, window_size):
    aaa = []
    for i in range(len(a)-window_size+1):
        subset = a[i:(i+window_size)]
        aaa.append([item for item in subset])
    print(type(aaa))
    return np.array(aaa)

dataset = split_5(a, window_size) # 5씩 잘랐으니 (5, 6)가 된다. // window_size+1만큼씩 잘라진다.
print("======================")

x_train = dataset[:,0:4]
y_train = dataset[:,4]

print(x_train)
x_train = np.reshape(x_train, (len(a)-window_size+1, 4, 1))

x_test = np.array([[[21],[22],[23],[24]], [[22],[23],[24],[25]], 
                  [[23],[24],[25],[26]], [[24],[25],[26],[27]]])
y_test = np.array([25, 26, 27, 28])

# print(x_train.shape)    # (6, 4, 1)
# print(y_train.shape)    # (6, )
# print(x_test.shape)     # (4, 4, 1)
# print(y_test.shape)     # (4, )

print(x_train)
print(y_train)

# 모델 구성하기
model = Sequential()

model.add(LSTM(20, input_shape=(4,1)))
# model.add(LSTM(32, input_shape=(4,1), return_sequences=True))
# model.add(LSTM(10, return_sequences=True))
# model.add(LSTM(10, return_sequences=True))
# model.add(LSTM(10, return_sequences=True))
# model.add(LSTM(10, return_sequences=True))
# model.add(LSTM(10, return_sequences=True))
# model.add(LSTM(10, return_sequences=True))
# model.add(LSTM(10))


# model.add(Dropout(0.2))
model.add(Dense(10, activation='relu'))
model.add(Dense(30))
model.add(Dense(10))
model.add(Dense(1))

model.summary()

model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mse'])

model.fit(x_train, y_train, epochs=50, batch_size=1, verbose=2)

loss, acc = model.evaluate(x_test, y_test)

y_predict = model.predict(x_test)
# y_predict2 = model.predict(x_train)

print('loss : ', loss)
print('acc : ', acc)
print('y_predict(x_test) : \n', y_predict)
# print('y_predict2(x_train) : \n', y_predict2)

# print(a)
