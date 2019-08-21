#keras11_ensemble.py

# 1. Data
import numpy as np

xxx1 = np.array([range(10), range(10,20)])
xxx2 = np.array([range(20,30), range(30,40)])
# xxx = np.concatenate((xxx1, xxx2), axis=0)
yyy1 = np.array([range(10), range(10,20)])
yyy2 = np.array([range(20,30), range(30,40)])

print(xxx1)
print(yyy1)
xxx1 = xxx1.transpose()
xxx2 = xxx2.transpose()
yyy1 = yyy1.transpose()
yyy2 = yyy2.transpose()
print(xxx1)
print(yyy1)

from sklearn.model_selection import train_test_split
x1_train, x1_test, y1_train, y1_test = train_test_split(xxx1, yyy1, test_size=0.2)
x2_train, x2_test, y2_train, y2_test = train_test_split(xxx2, yyy2, test_size=0.2)

# 2. Model
# from keras.models import Sequential
from keras.layers import Dense, Input
from keras.models import Sequential, Model
from keras.layers.merge import concatenate

# Model 1
input1 = Input(shape=(2,))                          # 모델1의 입력값은 2
dense1 = Dense(100, activation='relu')(input1)      # dense1 레이어에 100개의 노드 생성
# Model 2
input2 = Input(shape=(2,))                          # 모델2의 입력값은 2
dense2 = Dense(50, activation='relu')(input2)       # dense2 레이어에 50개의 노드 생성

merge1 = concatenate([dense1, dense2])

output_1 = Dense(10)(merge1)
output1 = Dense(2)(output_1)
output_2 = Dense(20)(merge1)
output2 = Dense(2)(output_2)

model = Model(inputs=[input1, input2], outputs=[output1, output2])

model.summary()

# 3. Train(Compile)
model.compile(loss='mse', optimizer='adam', metrics=['acc'])
model.fit([x1_train, x2_train], [y1_train, y2_train], epochs=50, batch_size=1, validation_split=0.25)

# 4. Evaluate / Predict
print(model.metrics_names)
loss, ouput1_loss, output2_loss, output1_acc, output2_acc = model.evaluate([x1_test, x2_test], [y1_test, y2_test], batch_size=1)
print("output1_acc: ", output1_acc)
print("output2_acc: ", output2_acc)
print("ouput1_loss: ", ouput1_loss)
print("output2_loss: ", output2_loss)
print("loss: ", loss)

# model.predict()의 입력값은 하나로 받기 때문에 리스트로 전달
y1_predict, y2_predict = model.predict([x1_test, x2_test])
print(y1_predict)
print(y2_predict)

# RMSE
from sklearn.metrics import mean_squared_error
def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))

print("RMSE1: ", RMSE(y1_test, y1_predict))
print("RMSE2: ", RMSE(y2_test, y2_predict))

# R2
from sklearn.metrics import r2_score
r2_y1_predict = r2_score(y1_test, y1_predict)
r2_y2_predict = r2_score(y2_test, y2_predict)
print("R2_1: ", r2_y1_predict)
print("R2_2: ", r2_y2_predict)

