#bostonReg.py

from practice import bostonData
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt


# Simple Linear Regression
sim_lr = LinearRegression()
sim_lr.fit(bostonData.x_train['RM'].values.reshape((-1, 1)), bostonData.y_train)
y_predict = sim_lr.predict(bostonData.x_test['RM'].values.reshape((-1, 1)))

print('Simple Linear Regression{:.4f}'.format(r2_score(bostonData.y_test, y_predict)))
print('Simple Linear Regression\nCoefficient:{:.4f}\nIntercept:{:.4f}'.format(sim_lr.coef_[0], sim_lr.intercept_))

plt.scatter(bostonData.x_test['RM'], bostonData.y_test, s=10, c='black')
plt.plot(bostonData.x_test['RM'], y_predict, c='red')
plt.legend(['Regression line', 'x_test'], loc='upper left')
plt.show()

# Multi Linear Regression
mlt_lr = LinearRegression()
mlt_lr.fit(bostonData.x_train.values, bostonData.y_train)
y_predict = mlt_lr.predict(bostonData.x_test.values)

print('Multi Linear Regression\nCoefficient: {}\nIntercept: {:.4f}'.format(mlt_lr.coef_, mlt_lr.intercept_))
print('Multi Linear Regression\nR2: {:.4f}'.format(r2_score(bostonData.y_test, y_predict)))
