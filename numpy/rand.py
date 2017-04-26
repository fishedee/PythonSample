import numpy as np  

# 正态
mean = 1
sigma = 10
data = np.random.normal(mean,sigma,10)
print(data)

# 多维正态

mean = [0, 0]
cov = [[1, 0], [0, 100]]
data = np.random.multivariate_normal(mean, cov, 10)
print(data)