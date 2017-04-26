import numpy as np

# 创建array

a = np.array([1, 2, 3, 4])
b = np.array([[1, 2, 3, 4],[4, 5, 6, 7], [7, 8, 9, 10]])
print(a,b)

# shape与修改shape
print(a.shape,b.shape)
b.shape = 4,3
print(b)
b.shape = -1,2
print(b)

# ndim
print(a.ndim,b.ndim)

# dtype与修改dtype
print(a,a.dtype)
a = a.astype(np.float)
print(a,a.dtype)

# T 转置
c = np.array([[11,21],[12,22],[13,23]])
print(c,c.T)

# 填充array

print( np.arange(0,1,0.1))
print( np.linspace(0,1,12))
print( np.fromfunction(lambda i,j:(i+1) * (j+1), (9,9)))

# 访问与设置array
a = np.array([101,4,5,6])
print( a ,a[0:-1],a[0])
a[0] = 3
print( a )

print(a[[0,2]])
a[[0,2]] = 7,8
print(a)

# 多维array

d = np.arange(0, 60, 10).reshape(-1, 1) + np.arange(0, 6)
print(d)
print(d[(0,1,2,3,4),(1,2,3,4,5)])
print(d[3:, [0, 2, 5]])

# 遍历array
d = np.arange(0, 60, 10).reshape(-1, 1) + np.arange(0, 6)
for i in d:
	for j in i:
		print(j,end=",")
	print("")

#转换list
print( d.tolist() )
