#keras10_mlp.py

# 1. Data
import numpy as np

# xxx = np.array([range(100), range(311,411)])
# yyy = np.array([range(501,601), range(611,711)])

xxx = np.array([range(10), range(10,20)])
yyy = np.array([range(10), range(10,20)])

# print(xxx)
# print(xxx.shape)
# xxx = np.transpose(xxx)
# yyy = np.transpose(yyy)
# xxx = xxx.reshape(10,2)       # reshape함수는 10행 2열을 2행 10열로 바꿀 수는 있지만 x, y 짝으로 이어지지 않는다.
# yyy = yyy.reshape(10,2)

xxx = xxx.transpose()
yyy = yyy.transpose()

# print(xxx)
# print(xxx.shape)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(xxx, yyy, test_size=0.2)

# 2. Model
from keras.models import Sequential
from keras.layers import Dense

model = Sequential()
# model.add(Dense(5, input_dim=2, activation='relu'))
model.add(Dense(5, input_shape=(2,), activation='relu'))
model.add(Dense(2))
model.add(Dense(5))
model.add(Dense(2))
model.add(Dense(2))

# 3. Train(Compile)
model.compile(loss='mse', optimizer='adam', metrics=['acc'])
model.fit(x_train, y_train, epochs=250, batch_size=1, validation_split=0.25)

# 4. Evaluate / Predict
loss, acc = model.evaluate(x_test, y_test, batch_size=1)
print("acc: ", acc)
print("loss: ", loss)

y_predict = model.predict(x_test)
print(y_predict)

# RMSE
from sklearn.metrics import mean_squared_error
def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))

print("RMSE: ", RMSE(y_test, y_predict))

# R2
from sklearn.metrics import r2_score
r2_y_predict = r2_score(y_test, y_predict)
print("R2: ", r2_y_predict)
