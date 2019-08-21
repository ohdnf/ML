import numpy as np

xxx = np.array([range(10), range(10,20), range(20,30)])
yyy = np.array([range(10), range(10,20), range(20,30)])

print(xxx)
print(xxx.shape)

# xxx = xxx.reshape()
# xxx = xxx.transpose()
xxx = xxx.swapaxes(0,1)

print(xxx)
print(xxx.shape)