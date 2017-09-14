import numpy as np
import math 

# 普通的全梯度下降+xor方法

#加入训练数据

x_train = np.array([[0,1],[1,0],[0,0],[1,1]])
y_train = np.array([[1],[1],[0],[0]])

# 训练
l1_weight = np.random.random((2,3))
l2_weight = np.random.random((1,3))
l1_grade = np.zeros((2,3))
l2_grade = np.zeros((1,3))
l1_last_grade = np.zeros((2,3))
l2_last_grade = np.zeros((1,3))
rate = 0.1
discount = 0.9

def sigmoid(x):
	return 1.0/(1.0+np.exp(-x))

def grad_sigmoid(x):
	return np.exp(-x)/((1+np.exp(-x))**2)

def add_one(x):
	return np.vstack((x,np.array([1])))

def fit(x,y):
	global l1_weight,l2_weight,l1_grade,l2_grade

	# 前向计算
	l1_input = x.reshape((2,1))
	l1s_input = np.dot(l1_weight,add_one(l1_input))
	l2_input = sigmoid(l1s_input)
	l2s_input = np.dot(l2_weight,add_one(l2_input))
	output = sigmoid(l2s_input)
	
	# 后向计算output的loss
	output_loss = output-y

	# 后向计算l2s的loss
	l2s_loss = output_loss*grad_sigmoid(l2s_input).T

	# 后向计算l2的loss和梯度
	l2_grade += (np.dot(add_one(l2_input),l2s_loss)).T
	l2_loss = np.dot(l2s_loss,l2_weight)
	l2_loss = l2_loss[:,:-1]

	# 后向计算l1s的loss
	l1s_loss = l2_loss*grad_sigmoid(l1s_input).T

	# 计算l1的loss和梯度
	l1_grade += (np.dot(add_one(l1_input),l1s_loss)).T
	l1_loss = np.dot(l1s_loss,l1_weight)
	l1_loss = l1_loss[:-1]

	return (output_loss)**2

def updateGrade():
	global l1_weight,l2_weight,l1_grade,l2_grade,l1_last_grade,l2_last_grade

	l1_last_grade = l1_last_grade*discount+l1_grade
	l1_weight = l1_weight - rate*l1_last_grade

	l2_last_grade = l2_last_grade*discount+l2_grade
	l2_weight = l2_weight - rate*l2_last_grade

	l1_grade = np.zeros((2,3))
	l2_grade = np.zeros((1,3))

def predict(x):
	global l1_weight,l2_weight

	l1_input = x.reshape((2,1))
	l1s_input = np.dot(l1_weight,add_one(l1_input))
	l2_input = sigmoid(l1s_input)
	l2s_input = np.dot(l2_weight,add_one(l2_input))
	output = sigmoid(l2s_input)

	return output

for epoch in range(0,10000):

	# 倒入数据
	allLoss = 0
	for i in range(0,len(x_train)):
		allLoss += fit(x_train[i],y_train[i])

	# 更新梯度
	updateGrade()

	print("epoch: %d-loss: %f"%(epoch,allLoss))

print(l1_weight)
print(l2_weight)

for i in range(0,len(x_train)):
	print("data: (%d,%d),predict:%f"%(x_train[i,0],x_train[i,1],predict(x_train[i])))
