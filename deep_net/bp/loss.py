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

	def get(self,predictOutput,realOutput):
		if self._type == "mse":
			return self._mse(predictOutput,realOutput)
		elif self._type == "binary_crossentropy":
			return self._binary_crossentropy(predictOutput,realOutput)
		else:
			raise Exception("Unknown Loss Type : %s"%(self._type))