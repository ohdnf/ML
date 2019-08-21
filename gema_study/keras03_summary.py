#keras03_summary.py

# 1. Data
import numpy as np
# Training Data
x = np.array([1,2,3,4,5,6,7,8,9,10])
y = np.array([1,2,3,4,5,6,7,8,9,10])
# Test Data
i = np.array([1,2,3,4,5,6,7,8,9,10])
j = np.array([1,2,3,4,5,6,7,8,9,10])

# 2. Model
from keras.models import Sequential
from keras.layers import Dense

model = Sequential()

model.add(Dense(5, input_dim=1, activation='relu'))
model.add(Dense(3))
model.add(Dense(1))

model.summary()

'''
# 3. Train(Compile)
model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])
model.fit(x, y, epochs=16, batch_size=1)

# 4. Evaluate / Predict
loss, acc = model.evaluate(i, j, batch_size=1)
print("acc: ", acc)
print("loss: ", loss)
'''