#keras07_R2.py

# 1. Data
import numpy as np

# Training Data
x_train = np.array([1,2,3,4,5,6,7,8,9,10])
x_test = np.array([1001,1002,1003,1004,1005,1006,1007,1008,1009,1020])
# Test Data
y_train = np.array([1,2,3,4,5,6,7,8,9,10])
y_test = np.array([1001,1002,1003,1004,1005,1006,1007,1008,1009,1020])

# 2. Model
from keras.models import Sequential
from keras.layers import Dense

model = Sequential()
model.add(Dense(5, input_dim=1, activation='relu'))
model.add(Dense(2))
model.add(Dense(1))

# 3. Train(Compile)
model.compile(loss='mse', optimizer='adam', metrics=['acc'])
# model.compile(loss='mse', optimizer='adam')
# model.compile(loss='mae', optimizer='adam', metrics=['mae'])
model.fit(x_train, y_train, epochs=250, batch_size=1)

# 4. Evaluate / Predict
loss, acc = model.evaluate(x_test, y_test, batch_size=1)
print("acc: ", acc)
print("loss: ", loss)

y_predict = model.predict(x_test)

# RMSE 구하기
from sklearn.metrics import mean_squared_error
def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))

# R2
from sklearn.metrics import r2_score
r2_y_predict = r2_score(y_test, y_predict)
print("R2: ", r2_y_predict)
