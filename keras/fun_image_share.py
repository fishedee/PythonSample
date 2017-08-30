from keras.layers import Input, Embedding, LSTM, Dense,Conv2D,MaxPooling2D,Flatten
from keras.models import Model
import numpy as np
import keras as keras

# 生成随机数据，一堆图片，以及label标签
img1_train = np.random.random((1000, 27, 27,3))
img2_train = np.random.random((1000, 27, 27,3))
label_train = np.random.randint(2, size=(1000, 1))

shareVisualModel = None

def getShareVisualModel():
	global shareVisualModel
	if shareVisualModel is not None:
		return shareVisualModel
	# 输入数据为(1000,27,27,3)
	digit_input = Input(shape=(27, 27,3))

	# 卷积层，数据为(1000,27,27,64)
	x = Conv2D(64, (3, 3))(digit_input)

	# 卷积层，数据为(1000,27,27,64)
	x = Conv2D(64, (3, 3))(x)

	# 抽样层，数据为(1000,13,13,64)
	x = MaxPooling2D((2, 2))(x)

	# 拍扁，数据为(1000,10816)
	out = Flatten()(x)
	shareVisualModel = Model(digit_input, out)
	return shareVisualModel

def leftVisualModel():
	digit_a = Input(shape=( 27, 27,3))
	return digit_a,getShareVisualModel()(digit_a)

def rightVisualModel():
	digit_a = Input(shape=( 27, 27,3))
	return digit_a,getShareVisualModel()(digit_a)

def globalModel():
	leftInput,leftOutput = leftVisualModel()
	rightInput,rightOutput = rightVisualModel()
	# 叠加层，叠加方式为维度扩容，数据为数据为(1000,21632)
	concatenated = keras.layers.concatenate([leftOutput, rightOutput])

	#单分类层
	out = Dense(1, activation='sigmoid')(concatenated)
	return [leftInput,rightInput],out

inputs,outputs = globalModel()
model = Model(inputs=inputs,outputs=outputs)
print(model.summary())
model.compile(optimizer='rmsprop', loss='binary_crossentropy')
model.fit([img1_train,img2_train], [label_train], epochs=50, batch_size=32)