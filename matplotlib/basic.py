import numpy as np
import matplotlib.pyplot as plt

# figure容器

x = np.linspace(-100, 100, 1000)

plt.figure(figsize=(8,4))#800x400的区域
plt.plot(x,x**2)#散点图
plt.xlabel("x")#横轴标题
plt.ylabel("$x^2$")#竖轴标题
plt.title("$y = x^2$")#标题
plt.xlim(-100,100)#横轴范围
plt.ylim(-10000,10000)#竖轴范围
plt.show()

# axes容器之subplot
plt.figure(figsize=(8,4))
plt.subplot(221) # 2行2列的第1个
plt.subplot(222) # 2行2列的第2个
plt.subplot(223) # 2行2列的第3个
plt.subplot(224) # 2行2列的第4个
plt.show()

# axes容器之axes
fig = plt.figure(figsize=(8,4))
fig.add_axes([0.15, 0.1, 0.7, 0.3])
plt.show()

# axis容器
fig = plt.figure(figsize=(8,4))
ax = fig.add_axes([0.15, 0.1, 0.7, 0.3])
for tick in ax.xaxis.get_major_ticks():
    tick.label1.set_fontsize(30)
plt.show()
