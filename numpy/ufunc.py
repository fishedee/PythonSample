import numpy as np

#单目运算

a = np.array([1, 2, 3, 4])
print( np.sin(a) )
print(-a)

double_ufunc = np.frompyfunc( lambda x: x* 2, 1, 1)
print(double_ufunc(a))

# 双目运算

a = np.array([1, 2, 3, 4])
b = np.array([11, 21, 31, 41])

print(a+b,a-b,a*b,a/b,a%b,a**b)

# 双目运算的聚合方法

print (np.add.reduce([1,2,3]))
print (np.add.reduce([[1,2,3],[4,5,6]], axis=0))
print (np.add.reduce([[1,2,3],[4,5,6]], axis=1))
print (np.add.reduce([[1,2,3],[4,5,6]], axis=(0,1)))

# 不同shape的双目运算广播法则

a = np.arange(0, 60, 10).reshape(-1, 1)
b = np.arange(0, 5)
print(a,b,a+b)

x,y = np.ogrid[0:60:10,0:5:1]
print(x,y,x*y)
