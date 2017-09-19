import numpy as np
class Activation:
	class sigmoid:
		def __init__(self):
			pass
		def get_output(self,weight,inData):
		 	return 1.0/(1.0+np.exp(-inData))
		def get_loss(self,nextLoss,weight,inData,outData):
		 	return nextLoss*np.exp(-inData)/((1+np.exp(-inData))**2)

	class relu:
		def __init__(self):
			pass
		def get_output(self,weight,inData):
			result = inData.copy()
			result[inData<0]=0
			return result
		def get_loss(self,nextLoss,weight,inData,outData):
			result = inData.copy()
			result[inData<0]=0
			result[inData>=0]=1
			return nextLoss*result

	def __init__(self,type,input_shape=None):
		if type == "sigmoid":
			self._handler = self.sigmoid()
		elif type == "relu":
			self._handler = self.relu()
		else:
			raise Exception("Unknown Activation Type: %s"%(self._type))
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
		return self._handler.get_output(weight,inData)
	def get_loss(self,nextLoss,weight,inData,outData):
		return self._handler.get_loss(nextLoss,weight,inData,outData)
	def get_grade(self,nextLoss,weight,inData,outData):
		return np.array([])