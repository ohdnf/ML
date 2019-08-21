#keras06_RMSE.py

# 1. Data
import numpy as np

# Training Data
x_train = np.array([1,2,3,4,5,6,7,8,9,10])
x_test = np.array([11,12,13,14,15,16,17,18,19,20])
# Test Data
y_train = np.array([1,2,3,4,5,6,7,8,9,10])
y_test = np.array([11,12,13,14,15,16,17,18,19,20])

# 2. Model
from keras.models import Sequential
from keras.layers import Dense

model = Sequential()
model.add(Dense(5, input_dim=1, activation='relu'))
model.add(Dense(2))
model.add(Dense(1))

# 3. Train(Compile)
model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])
model.fit(x_train, y_train, epochs=100, batch_size=1)

# 4. Evaluate / Predict
loss, acc = model.evaluate(x_test, y_test, batch_size=1)
print("acc: ", acc)
print("loss: ", loss)

y_predict = model.predict(x_test)

# RMSE 구하기
from sklearn.metrics import mean_squared_error
def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))

print("RMSE: ", RMSE(y_test, y_predict))
