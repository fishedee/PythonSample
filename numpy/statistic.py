import numpy as np  

# 单一数据统计
mean = 1
sigma = 10
data = np.random.normal(mean,sigma,1000000)
print(np.mean(data),np.median(data),np.min(data),np.max(data))
print(np.var(data),np.std(data))

# 多数据统计

mean = [9, 4]
cov = [[10, 15], [15, 100]]
data = np.random.multivariate_normal(mean, cov, 1000000)
print(np.mean(data,axis=0),np.median(data,axis=0),np.min(data,axis=0),np.max(data,axis=0))
print(np.var(data,axis=0),np.std(data,axis=0),np.cov(data.T),np.corrcoef(data.T))