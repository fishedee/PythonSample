import numpy as np
import matplotlib.pyplot as plt

row = 3
col = 3
n = 1

plt.figure()

# 折线图
x = np.linspace(0, 2*np.pi, 50)

plt.subplot(row,col,n)
n+=1
plt.plot(x,np.sin(x))

plt.subplot(row,col,n)
n+=1
plt.plot(x,np.sin(x),"--")

plt.subplot(row,col,n)
n+=1
plt.plot(x,np.sin(x),"+")

# 直方图
plt.subplot(row,col,n)
n+=1
plt.hist(np.random.normal(5,10,100000))  

plt.subplot(row,col,n)
n+=1
plt.hist(np.random.normal(5,10,100000),100)  

plt.subplot(row,col,n)
n+=1
plt.hist(np.random.normal(5,10,100000),100,normed=True)  



# 散点图
x = np.linspace(0, 2*np.pi, 10)

plt.subplot(row,col,n)
n+=1
plt.scatter(x,np.cos(x),100)  

plt.subplot(row,col,n)
n+=1
plt.scatter(x,np.cos(x),100,marker='x')  

plt.subplot(row,col,n)
n+=1
plt.scatter(x,np.cos(x),100,marker='o')  

plt.show()