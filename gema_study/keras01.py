#keras01.py

# 1. Data
import numpy as np
# Training Data
x = np.array([2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 
53, 59, 61, 67, 71, 73, 79, 83, 89, 97])
y = np.array([2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 
53, 59, 61, 67, 71, 73, 79, 83, 89, 97])
# Test Data
i = np.array([1009, 1013, 1019, 1021, 1031, 1033, 1039, 
1049, 1051, 1061, 1063, 1069, 1087, 1091, 1093, 1097])
j = np.array([1009, 1013, 1019, 1021, 1031, 1033, 1039, 
1049, 1051, 1061, 1063, 1069, 1087, 1091, 1093, 1097])

# 2. Model
# keras에서 Sequential과 Dense 가져오기
from keras.models import Sequential
from keras.layers import Dense

model = Sequential()

# input node가 하나, output node가 둘인 Layer 생성
model.add(Dense(2, input_dim=1, activation='relu'))
# input node가 둘, output node가 셋인 Layer 생성
model.add(Dense(3))
# input node가 셋, output node가 넷인 Layer 생성
model.add(Dense(4))
# input node가 넷, output node가 하나인 Layer 생성
model.add(Dense(1))

# 3. Train(Compile)
model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])
model.fit(x, y, epochs=150, batch_size=1)

# 4. Evaluate / Predict
loss, acc = model.evaluate(i, j, batch_size=1)
print("acc: ", acc)
print("loss: ", loss)