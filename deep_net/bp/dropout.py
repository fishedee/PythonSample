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
		total = self._input_shape[1]
		oneTotal = total * closePor
		zeroTotal = total - oneTotal
		oneData = [1]*oneTotal
		zeroData = [0]*zeroTotal
		data = np.array(oneData+zeroData)
		self._drop = np.random.shuffle(data)
		return inData * self._drop
	def get_loss(self,nextLoss,weight,inData,outData):
		return nextLoss* self._drop
	def get_grade(self,nextLoss,weight,inData,outData):
		return np.array([])