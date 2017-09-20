import numpy as np
class Dropout:
	def __init__(self,closePor,input_shape=None):
		self._closePor = closePor
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
		self._drop = np.random.random(inData.shape)
		self._drop[self._drop<self._closePor] = 0
		self._drop[self._drop>=self._closePor] = 1
		return inData * self._drop
	def get_loss(self,nextLoss,weight,inData,outData):
		return nextLoss* self._drop
	def get_grade(self,nextLoss,weight,inData,outData):
		return np.array([])