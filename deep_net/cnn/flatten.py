import numpy as np
class Flatten:
	def __init__(self,unit,input_shape=None):
		self._unit = unit
		self._input_shape = input_shape
		self._setOutputShape()
	def _setOutputShape():
		if self._input_shape is not None:
			result = 1
			for i in range(0,len(self._input_shape)):
				result *= self._input_shape[i]
			self._output_shape = (1,result)
	def set_input_shape(self,input_shape):
		self._input_shape = input_shape
		self._setOutputShape()
	def get_output_shape(self):
		return self._output_shape
	def get_init_weight(self):
		return np.array([])
	def get_init_grade(self):
		return np.array([])
	def get_output(self,weight,inData):
		return inData.reshape(self._output_shape)
	def get_loss(self,nextLoss,weight,inData,outData):
		return nextLoss.reshape(self._input_shape)
	def get_grade(self,nextLoss,weight,inData,outData):
		return np.array([])