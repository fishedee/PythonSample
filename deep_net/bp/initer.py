import numpy as np
import math
class Initer:
	def __init__(self,type="xavier"):
		self._type = type
	def get(self,n1,n2,size):
		width = math.sqrt(6/(n1+n2+1))
		return np.random.uniform(-width,width,size)