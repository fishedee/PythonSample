import numpy as np
class Activation:
	def __init__(self,type,input_shape=None):
		self._type = type
		self._input_shape = input_shape
	def set_input_shape(self,input_shape):
		self._input_shape = input_shape
	def get_output_shape(self):
		return self._input_shape
	def get_init_weight(self):
		return np.array([])
	def get_init_grade(self):
		return np.array([])
	def get_output(self,weight,inData):
		if self._type == "sigmoid":
			return 1.0/(1.0+np.exp(-inData))
		elif self._type == "relu":
			result = inData.copy()
			result[inData<0]=0
			return result
		else:
			raise Exception("Unknown Activation Type: %s"%(self._type))
	def get_loss(self,nextLoss,weight,inData):
		if self._type == "sigmoid":
			return nextLoss*np.exp(-inData)/((1+np.exp(-inData))**2)
		elif self._type == "relu":
			result = inData.copy()
			result[inData<0]=0
			result[inData>=0]=1
			return nextLoss*result
		else:
			raise Exception("Unknown Activation Type: %s"%(self._type))
	def get_grade(self,nextLoss,weight,inData):
		return np.array([])