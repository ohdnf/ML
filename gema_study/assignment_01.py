#assignment_01.py

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import mode
import statistics as sta

#1/5
def npa() :
    npa = np.array([[1,2,3],[4,5,6],[7,8,9]])
    print(npa.size)
    print(npa.shape)
    print(len(npa))

def npl() :
    npl = np.array([1, 100, 42, 42, 42, 6, 7])
    print(npl.size)
    print(len(npl))
    print(npl.shape)
    print(npl.ndim)

def zero() :
    a = np.zeros(3)
    print(a.ndim)
    print(a.shape)
    print(a)

def eye():
    np.eye(2,dtype = int)
    print(np.eye(3))
    print(np.eye(3,k=1))
    print(np.eye(3,k=-1))

def identity() :
    print(np.identity(5))

def cal1() :
    x = np.array([1,5,2])
    y = np.array([7,4,1])
    print(x+y)
    print(x*y)
    print(x-y)
    print(x/y)
    print(x%y)

def cal2() :
    bb = np.array([1,2,3])
    cc = np.array([-7,8,9])
    print(np.dot(bb,cc))

    xs = np.array(((2,3),(3,5)))
    ys = np.array(((1,2),(5,1)))
    print(np.dot(xs,ys),type(np.dot(xs,ys)))


def ndarray3() :
    outcome = np.random.randint(1,7,size = 10)
    print(outcome)
    print(type(outcome))
    print(len(outcome))

    print(np.random.randint(2, size =10))
    print(np.random.randint(1,size=10))
    print(np.random.randint(5,size=(2,4)))

def ndarray4() :
    a = np.random.randn(3,2) # 이거 달아주셈 import matplotlib.pyplot as plt
    print(a)
    b = np.random.randn(3,3,3)
    print(b)
    plt.plot(a)
    plt.show()
def ndarray5() :
    arr = np.arange(10)
    print(arr)
    np.random.shuffle(arr)
    print(arr)

    arr2 = np.arange(9).reshape((-1,3))
    print(arr2)
    np.random.shuffle(arr2)
    print(arr2)

def basic_stat1() :
    x = np.array([-2.1,-1,1,1,4.3])
    print(np.mean(x))
    print(np.median(x))
    print(mode(x))

def basic_stat2() :
    x = np.array([-2.1,-1,1,1,4.3])
    print(np.mean(x))
    print(np.median(x))
    print(mode(x))

    x_m = np.mean(x)
    x_a = x-x_m
    x_p = np.power(x_a,2)

    print("Variance x")
    print("np.var(x)")
    print(sta.pvariance(x))
    print(sta.variance(x))

if __name__ =='__main__':
    # npa()
    # npl()
    # zero()
    # eye()
    # identity()
    # cal1()
    # cal2()



    # ndarray3()
    # ndarray4()
    # ndarray5()
    # basic_stat1()
    basic_stat2()