#keras08_val.py

# 1. Data
import numpy as np

# import tensorflow as tf
# seed = 0
# np.random.seed(seed)
# tf.set_random_seed(seed)

# Training Data
# x_train = np.array([1,2,3,4,5,6,7,8,9,10])
x_train = np.array([1,2,3,6,7,8,9,10])
# y_train = np.array([1,2,3,4,5,6,7,8,9,10])
y_train = np.array([1,2,3,6,7,8,9,10])
# Test Data
x_test = np.array([1001, 1002, 1004, 1005, 1006, 1008, 1009, 1010])
y_test = np.array([1001, 1002, 1004, 1005, 1006, 1008, 1009, 1010])
# Validation Data
x_val = np.array([101, 103, 105, 106, 107, 108, 109, 110])
y_val = np.array([101, 103, 105, 106, 107, 108, 109, 110])

# 2. Model
from keras.models import Sequential
from keras.layers import Dense

model = Sequential()
model.add(Dense(5, input_dim=1, activation='relu'))
model.add(Dense(2))
model.add(Dense(5))
model.add(Dense(2))
model.add(Dense(1))

# 3. Train(Compile)
model.compile(loss='mse', optimizer='adam', metrics=['acc'])
model.fit(x_train, y_train, epochs=250, batch_size=1, 
          validation_data=(x_val, y_val))

# 4. Evaluate / Predict
loss, acc = model.evaluate(x_test, y_test, batch_size=1)
print("acc: ", acc)
print("loss: ", loss)

y_predict = model.predict(x_test)
print(y_predict)

# RMSE 구하기
from sklearn.metrics import mean_squared_error
def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))

# R2
from sklearn.metrics import r2_score
r2_y_predict = r2_score(y_test, y_predict)
print("R2: ", r2_y_predict)
