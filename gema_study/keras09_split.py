#keras09_split.py

# 1. Data
import numpy as np

xxx = np.array(range(100))
yyy = np.array(range(100))

# x_train = xxx[:60]
# x_val = xxx[60:80]
# x_test = xxx[80:]
# y_train = yyy[:60]
# y_val = yyy[60:80]
# y_test = yyy[80:]

# x_train, x_val, x_test = np.split(xxx, [60, 80])
# y_train, y_val, y_test = np.split(yyy, [60, 80])

# print("x_train.shape", x_train.shape)
# print("x_val.shape", x_val.shape)
# print("x_test.shape", x_test.shape)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(xxx, yyy, 
    test_size=0.2, #random_state=55,
)

# print("x_train \n%s" % x_train)
# print("y_train \n%s" % y_train)
# print("x_test \n%s" % x_test)
# print("y_test \n%s" % y_test)

# 2. Model
from keras.models import Sequential
from keras.layers import Dense

model = Sequential()
# model.add(Dense(5, input_dim=1, activation='relu'))
model.add(Dense(5, input_shape=(1,), activation='relu'))
model.add(Dense(2))
model.add(Dense(5))
model.add(Dense(2))
model.add(Dense(1))

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

# R2
from sklearn.metrics import r2_score
r2_y_predict = r2_score(y_test, y_predict)
print("R2: ", r2_y_predict)
