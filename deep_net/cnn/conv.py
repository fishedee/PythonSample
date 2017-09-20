import numpy as np
from initer import Initer
class Conv2D:
	def __init__(self,unit,kernal_size,input_shape=None):
		self._unit = unit
		self._kernal_shape = kernal_size
		self._input_shape = input_shape 
		self._setOutputShape()
	def _setOutputShape():
		if self._input_shape is not None:
			channel = self._unit
			height = self._input_shape[1]-self._kernal_shape[0]+1
			width = self._input_shape[2]-self._kernal_shape[1]+1
			self._output_shape = (channel,height,width)
			self._weight_shape = (self,_unit,self.input_shape[0])+self._kernal_shape
	def _convolve(a,b):
		return np.convolve(a,b.T,mode="valid")
	def set_input_shape(self,input_shape):
		self._input_shape = input_shape
		self._setOutputShape()
	def get_output_shape(self):
		return self._output_shape
	def get_init_weight(self):
		inDataNum = self._input_shape[0]*self._input_shape[1]*self._input_shape[2]
		outDataNum = self._output_shape[0]*self._output_shape[1]*self._output_shape[2]
		initer = Initer()
		return initer.get(inDataNum,outDataNum,self._weight_shape)
	def get_init_grade(self):
		return np.zeros(self._weight_shape)
	def get_output(self,weight,inData):
		result = np.zeros(self._output_shape)
		for i in range(0,len(weight)):
			for j in range(0,len(weight[i])):
				result[i] = result[i]+self._convolve(inData[i,j],weight[i,j])

		return result
	def get_loss(self,nextLoss,weight,inData,outData):
		#填充0值
		height = self._kernal_shape[0]
		width = self._kernal_shape[1]
		npad = ((0, 0), (width-1, width-1), (height-1, height-1))
		nextLoss = np.pad(nextLoss, pad_width=npad, mode='constant', constant_values=0)

		#计算
		result = np.zeros(inData.shape)
		for i in range(0,len(weight)):
			for j in range(0,len(weight[i])):
				result[j] = result[j]+self._convolve(nextLoss[i],weight[i,j])

		return result
	def get_grade(self,nextLoss,weight,inData,outData):
		result = np.zeros(self._weight_shape)

		for i in range(0,len(inData)):
			for j in range(0,len(nextLoss)):
				result[j,i]=self._convolve(inData[i],nextLoss[j])

		return result