#keras15_cnn_model.py
from keras.models import Sequential
from keras.layers import Conv2D, Flatten, MaxPooling2D

filter_size = 32        # 사용자 임의로 지정
kernel_size = (3,3)     # 사용자 임의로 지정 / Square vs. Rectangle
model = Sequential()

model.add(Conv2D(32, (2,2), input_shape=(7,7,1), padding='same'))
#padding='same'은 가장자리에 0을 넣고 실행 따라서 shape이 (n-1, n-1)되지 않는다.
model.add(Conv2D(16, (3,3)))
model.add(MaxPooling2D(pool_size=2))
model.add(Flatten())
# Flatten되었으니 Dense 사용
model.summary()

'''
model.add(Conv2D(filter_size, kernel_size, #padding = 'valid',
                 input_shape=(28,28,1)))
model.add(Conv2D(16, (3,3)))

from keras.layers import MaxPooling2D
pool_size = (2,2)
model.add(MaxPooling2D(pool_size))

model.summary()
'''