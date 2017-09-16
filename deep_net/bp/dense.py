import numpy as np
class Dense:
	def __init__(self,unit,input_shape=None):
		self._unit = unit
		self._input_shape = input_shape 
	def _add_one(self,data):
		return np.hstack((data,np.array([[1]])))
	def set_input_shape(self,input_shape):
		self._input_shape = input_shape
	def get_output_shape(self):
		return (self._unit,)
	def get_init_weight(self):
		inputShape = self._input_shape[0]+1
		return np.random.random((inputShape,self._unit))
	def get_init_grade(self):
		inputShape = self._input_shape[0]+1
		return np.zeros((inputShape,self._unit))
	def get_output(self,weight,inData):
		return np.dot(self._add_one(inData),weight)
	def get_loss(self,nextLoss,weight):
		loss = np.dot(weight,nextLoss)
		return loss[:-1,]
	def get_grade(self,nextLoss,inData):
		return np.dot(nextLoss,self._add_one(inData)).T