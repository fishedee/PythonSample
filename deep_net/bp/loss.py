import numpy as np
class Loss:
	def __init__(self,type):
		self._type = type
	def _mse(self,predictOutput,realOutput):
		if realOutput is None:
			return predictOutput,np.zeros(predictOutput.shape),0,0
		predictOutput = predictOutput.T
		realOutput = realOutput.T
		predictOutputNorm = np.linalg.norm(predictOutput,2)
		realOutputNorm = np.linalg.norm(realOutput,2)

		predictY = predictOutput
		layerLoss = predictOutput - realOutput
		loss = np.dot(layerLoss,layerLoss.T)*0.5
		acc = 0
		return predictY,layerLoss,loss,acc
	def get(self,predictOutput,realOutput):
		if self._type == "mse":
			return self._mse(predictOutput,realOutput)
		else:
			raise Exception("Unknown Loss Type : %s"%(self._type))