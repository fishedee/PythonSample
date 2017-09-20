import numpy as np
def to_categorical(y,num_classes):
	yShape = y.shape
	classShape = (num_classes,)
	resultShape = yShape[0:-1]+classShape
	result = np.zeros(resultShape)
	resultFlat = result.flat
	yFlat = y.flat
	for i in range(0,len(yFlat)):
		resultFlat[i*num_classes+yFlat[i]]=1
	return result