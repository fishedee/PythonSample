import numpy as np
class MaxPooling2D:
	def __init__(self,unit,pool_size,input_shape=None):
		self._unit = unit
		self._kernal_shape = pool_size
		self._input_shape = input_shape
		self._setOutputShape()
	def _setOutputShape():
		if self._input_shape is not None:
			channels = self._input_shape[0]
			height = int(self._input_shape[1]/self._kernal_shape[0])
			width = int(self._input_shape[2]/self._kernal_shape[1])
			self._output_shape = (channels,height,width)
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
		result = np.zeros(self._output_shape)
		self._maxPos = [[],[],[]]
		strideHeight = self._kernal_shape[0]
		strideWidth = self._kernal_shape[1]
		for channel in range(0,len(result))
			for i in range(0,len(result[channel])):
				for j in range(0,len(result[channel,i])):
					maxDatai = None
					maxDataj = None
					begini = strideHeight * i
					beginj = strideWidth * j
					for ki in range(0,strideHeight):
						for kj in range(0,strideWidth):
							if maxDatai is None:
								maxDatai = begini+ki
								maxDataj = beginj+kj
							elif inData[channel,maxDatai,maxDataj] < inData[channel,begini+ki,beginj+kj]:
								maxDatai = begini+ki
								maxDataj = beginj+kj
					self._maxPos[0].append(channel)
					self._maxPos[1].append(maxDatai)
					self._maxPos[2].append(maxDataj)
					result[channel,i,j] = inData[channel,maxDatai,maxDataj]

		return result
	def get_loss(self,nextLoss,weight,inData,outData):
		result = np.zeros(self._input_shape)
		result[self._maxPos[0],self._maxPos[1],self._maxPos[2]] = nextLoss.flatten()
		return result
	def get_grade(self,nextLoss,weight,inData,outData):
		return np.array([])