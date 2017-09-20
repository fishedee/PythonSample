import sys
sys.path.append("..")

import numpy as np
from bp.initer import Initer
class Dense:
	def __init__(self,unit,input_shape=None):
		self._unit = unit
		self._input_shape = input_shape 
	def _add_one(self,data):
		return np.hstack((data,np.array([[1]])))
	def set_input_shape(self,input_shape):
		self._input_shape = input_shape
	def get_output_shape(self):
		return (1,self._unit)
	def get_init_weight(self):
		inputShape = self._input_shape[1]+1
		initer = Initer()
		return initer.get(inputShape,self._unit,(inputShape,self._unit))
	def get_init_grade(self):
		inputShape = self._input_shape[1]+1
		return np.zeros((inputShape,self._unit))
	def get_output(self,weight,inData):
		return np.dot(self._add_one(inData),weight)
	def get_loss(self,nextLoss,weight,inData,outData):
		loss = np.dot(weight,nextLoss.T).T
		return loss[:,:-1]
	def get_grade(self,nextLoss,weight,inData,outData):
		return np.dot(nextLoss.T,self._add_one(inData)).T