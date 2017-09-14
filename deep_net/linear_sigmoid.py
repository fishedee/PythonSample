import numpy as np
import math 

def sigmoid(x):
	return 1.0/(1.0+math.exp(-x))

def grad_sigmoid(x):
	return math.exp(-x)/((1+math.exp(-x))**2)

# 随机梯度下降+sigmoid输出方法
sample = 100
num_input = 5

#加入训练数据
normalRand = np.random.normal(0,0.1,sample)
weight = [7,99,-1,-333,0.06]
x_train = np.random.random((sample, num_input))
y_train = np.zeros((sample,1))
for i in range (0,len(x_train)):
	total = 0
	for j in range(0,len(x_train[i])):
		total += weight[j]*x_train[i,j]
	y_train[i] = sigmoid(total+normalRand[i])

# 训练
weight = np.random.random(num_input+1)
rate = 0.05
batch = 32

def train(x_train,y_train):
	# 计算loss
	global weight,rate
	predictZ = np.zeros((len(x_train),))
	predictY = np.zeros((len(x_train,)))
	for i in range(0,len(x_train)):
		predictZ[i] = np.dot(x_train[i],weight[0:num_input])+weight[num_input]
		predictY[i] = sigmoid(predictZ[i])
	loss = 0
	for i in range(0,len(x_train)):
		loss += (predictY[i]-y_train[i])**2

	# 计算梯度并更新
	for i in range(0,len(weight)-1):
		grade = 0
		for j in range(0,len(x_train)):
			grade += (predictY[j]-y_train[j])*grad_sigmoid(predictZ[j])*x_train[j,i]
		weight[i] = weight[i] - rate*grade

	grade = 0
	for j in range(0,len(x_train)):
		grade += (predictY[j]-y_train[j])
	weight[num_input] = weight[num_input] - rate*grade
	return loss

for epoch in range(0,100):
	begin = 0
	while begin < len(x_train):
		end = begin+batch
		if end > len(x_train):
			end = len(x_train)
		loss = train(x_train[begin:end],y_train[begin:end])
		begin = end
	print("epoch: %d-loss: %f"%(epoch,loss))

print(weight)