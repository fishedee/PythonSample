import numpy as np
class Loss:
	def __init__(self,type):
		self._type = type
	def _mse(self,predictOutput,realOutput):
		if realOutput is None:
			return predictOutput,np.zeros(predictOutput.shape),0,0

		predictY = predictOutput
		layerLoss = predictOutput - realOutput
		loss = np.dot(layerLoss,layerLoss.T)*0.5
		acc = 0
		return predictY,layerLoss,loss,acc

	def _binary_crossentropy(self,predictOutput,realOutput):
		if predictOutput >= 0.5:
			predictY = 1
		else:
			predictY = 0
		if realOutput is None:
			return predictY,np.zeros(predictOutput.shape),0,0

		loss = -realOutput*np.log2(predictOutput)-(1-realOutput)*np.log2(1-predictOutput)
		layerLoss = -realOutput/predictOutput+(1-realOutput)/(1-predictOutput)
		if realOutput == predictY:
			acc = 1
		else:
			acc = 0

		return predictY,layerLoss,loss,acc

	def _categorical_crossentropy(self,predictOutput,realOutput):
		maxIndex = -1
		for i in range(0,len(predictOutput[0])):
			if maxIndex == -1:
				maxIndex = i
			elif predictOutput[0,i] > predictOutput[0,maxIndex]:
				maxIndex = i
		predictY = maxIndex
		if realOutput is None:
			return predictY,np.zeros(predictOutput.shape),0,0

		loss = -np.sum(realOutput*np.log2(predictOutput))
		layerLoss = realOutput/(-predictOutput)
		if loss < 1e-6:
			acc = 1
		else:
			acc = 0

		return  predictY,layerLoss,loss,acc

	def _softmax_categorical_crossentropy(self,predictOutput,realOutput):
		def get_output(inData):
			maxData = np.max(inData)
			inData -= maxData
			result = np.exp(inData)
			total = np.sum(result)
			return result/total
		def get_max_index(inData):
			maxIndex = -1
			for i in range(0,len(inData[0])):
				if maxIndex == -1:
					maxIndex = i
				elif inData[0,i] > inData[0,maxIndex]:
					maxIndex = i
			return maxIndex
		def get_layer_loss(predictData,realData):
			layerLoss = predictData.copy()
			for i in range(0,len(realData[0])):
				if realData[0,i] == 1:
					layerLoss[0,i] -= 1
			return layerLoss

		outpuData = get_output(predictOutput)
		predictY = get_max_index(outpuData)
		if realOutput is None:
			print(predictY)
			return predictY,np.zeros(predictOutput.shape),0,0

		dictance = outpuData-realOutput
		loss = np.dot(dictance,dictance.T)*0.5
		layerLoss = get_layer_loss(outpuData,realOutput)
		realY = get_max_index(realOutput)
		if predictY == realY:
			acc = 1
		else:
			acc = 0
		return  predictY,layerLoss,loss,acc

	def get(self,predictOutput,realOutput):
		if self._type == "mse":
			return self._mse(predictOutput,realOutput)
		elif self._type == "binary_crossentropy":
			return self._binary_crossentropy(predictOutput,realOutput)
		elif self._type == "categorical_crossentropy":
			return self._categorical_crossentropy(predictOutput,realOutput)
		elif self._type == "softmax_categorical_crossentropy":
			return self._softmax_categorical_crossentropy(predictOutput,realOutput)
		else:
			raise Exception("Unknown Loss Type : %s"%(self._type))