from model import *
from dense import *
from activation import *
from loss import *
from optimizer import *
import numpy as np
import math


# 线性+sigmoid

def sigmoid(x):
	return 1.0/(1.0+math.exp(-x))

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

model = Model()
model.add(Dense(1,input_shape=(5,)))
model.add(Activation("sigmoid"))

optimizer = Optimizer(rate=0.05,momentum=0.9)
loss = Loss("mse")
model.compile(optimizer=optimizer,loss=loss)

model.fit(x_train,y_train,epochs=1000)

print(model.get_weight())