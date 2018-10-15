import numpy as np
import datetime

#datetime转换为datetime64
now = datetime.datetime.now()
nextDay = datetime.datetime.now()+datetime.timedelta(days=1)
nextDay2 = datetime.datetime.now()+datetime.timedelta(days=2)

array1 = np.array([
	np.datetime64(now,'Y'),
	np.datetime64(nextDay,'Y'),
	np.datetime64(nextDay2,'Y'),
])

array2 = np.array([
	np.datetime64(now,'M'),
	np.datetime64(nextDay,'M'),
	np.datetime64(nextDay2,'M'),
])

array3 = np.array([
	np.datetime64(now,'D'),
	np.datetime64(nextDay,'D'),
	np.datetime64(nextDay2,'D'),
])


print(array1,type(array1),array1.dtype)
print(array2,type(array2),array2.dtype)
print(array3,type(array3),array3.dtype)

#运算
array4 = array3 + np.timedelta64(1, 'D')

print(array4,type(array4),array4.dtype)