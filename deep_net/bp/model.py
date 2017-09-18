class Model:
	def __init__(self):
		self._layer = []
		pass
	def add(self,layer):
		if len(self._layer) != 0:
			in_shape = self._layer[-1].get_output_shape()
			layer.set_input_shape(in_shape)
		self._layer.append(layer)
	def compile(self,optimizer,loss):
		self._optimizer = optimizer
		self._losser = loss
		self._weight = []
		for layer in self._layer:
			self._weight.append(layer.get_init_weight())
	def get_weight(self):
		return self._weight
	def _trainSingle(self,x,y,grade):
		if x.ndim == 1:
			x = x[:,None].T
		if y is not None and y.ndim == 1:
			y = y[:,None].T
		output = [0]*len(self._layer)
		loss = [0]*len(self._layer)

		# 前向传播
		for i in range(0,len(self._layer)):
			weight = self._weight[i]
			layer = self._layer[i]
			if i == 0:
				inData = x
			else:
				inData = output[i-1]
			output[i] = layer.get_output(weight,inData)

		# 计算结果
		predictY,layerLoss,totalLoss,totalAcc = self._losser.get(output[-1],y)

		# 后向传播loss和grade
		for i in range(len(self._layer)-1,-1,-1):
			weight = self._weight[i]
			layer = self._layer[i]
			if i == 0:
				preOutput = x
			else:
				preOutput = output[i-1]
			if i == len(self._layer)-1:
				nextLoss = layerLoss
			else:
				nextLoss = loss[i+1]
			loss[i] = layer.get_loss(nextLoss,weight,preOutput)
			grade[i] = grade[i] + layer.get_grade(nextLoss,weight,preOutput)

		return totalLoss,totalAcc,predictY
		
	def _train(self,x,y):
		grade = []
		for layer in self._layer:
			grade.append(layer.get_init_grade())
		# 训练数据
		totalLoss = 0
		totalAcc = 0
		for i in range(0,len(x)):
			loss,acc,predictY = self._trainSingle(x[i],y[i],grade)
			totalLoss += loss
			totalAcc += acc
		
		# 更新梯度
		self._weight = self._optimizer.update(self._weight,grade)
		return totalLoss,totalAcc
		
	def fit(self,x,y,epochs=10,batch_size=32):
		for epoch in range(0,epochs):
			totalLoss = 0
			totalAcc = 0
			begin = 0
			while begin < len(x):
				end = begin+batch_size
				if end > len(x):
					end = len(x)
				loss,acc = self._train(x[begin:end],y[begin:end])
				totalLoss += loss
				totalAcc += acc
				begin = end
			print("epoch: %d-loss: %f-acc: %f"%(epoch,totalLoss/len(x),totalAcc/len(x)))
	def evaluate(self,x,y):
		return self._train(x,y)
	def predict(self,x):
		grade = []
		for layer in self._layer:
			grade.append(layer.get_init_grade())
		loss,acc,predictY = self._trainSingle(x,None,grade)
		return predictY
