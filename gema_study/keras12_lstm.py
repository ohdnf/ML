#keras12_lstm.py
from numpy import array
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense

# 1. Data
X = array([[1,2,3],[2,3,4],[3,4,5],[4,5,6]])
y = array([4,5,6,7])

print("X.shape: ", X.shape)
print("y.shape: ", y.shape)

print(X)

# reshape from [samples, timesteps] into [samples, timesteps, features]
X = X.reshape((X.shape[0], X.shape[1], 1))

print("X.shape: ", X.shape)
print("y.shape: ", y.shape)

print(X)

# 2. Model
model = Sequential()
model.add(LSTM(50, activation='relu', input_shape=(3, 1)))      # (4, 3, 1) -> 행 무시 -> (3, 1)
# model.add(LSTM(25))       # LSTM을 두 개 연결하려면 옵션이 필요함
model.add(Dense(100))
model.add(Dense(40))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')

# 3. Train
model.fit(X, y, epochs=300, verbose=2)
# demonstrate prediction
# x_input = array([6, 7, 8])           # 60, 70, 80 => ?
x_input = array([60, 70, 80])
x_input = x_input.reshape((1, 3, 1))
yhat = model.predict(x_input, verbose=0)
print(yhat)
