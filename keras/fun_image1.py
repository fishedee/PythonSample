from keras.layers import Input, Embedding, LSTM, Dense,Conv2D,MaxPooling2D,Flatten
from keras.models import Model
import numpy as np
import keras as keras

# 生成随机数据，一堆图片，以及label标签
img_train = np.random.random((1000, 256, 256,3))
label_train = np.random.randint(2, size=(1000, 1))

shareInput = None

def getShareInput():
	# 共享输入图像
	global shareInput
	if shareInput is not None:
		return shareInput
	shareInput = Input(shape=(256, 256,3))
	return shareInput

def FirstConv2DModel():
	# 共享输入
	input_img =  getShareInput()

	# 卷积层，数据为(1000, 256,256,64)
	tower_1 = Conv2D(64, (1, 1), padding='same', activation='relu')(input_img)

	# 卷积层，数据为(1000, 256,256,64)
	tower_1 = Conv2D(64, (3, 3), padding='same', activation='relu')(tower_1)
	return input_img,tower_1

def SecondConv2DModel():
	input_img =  getShareInput()

	# 卷积层，数据为(1000, 256,256,64)
	tower_2 = Conv2D(64, (1, 1), padding='same', activation='relu')(input_img)

	# 卷积层，数据为(1000, 256,256,64)
	tower_2 = Conv2D(64, (5, 5), padding='same', activation='relu')(tower_2)
	return input_img,tower_2

def ThirdConv2DModel():
	input_img =  getShareInput()

	# 池化层，数据为(1000, 256,256,3)，注意步长为1，使得这里的长宽没有缩短，作用相当于让max传染到四周
	tower_3 = MaxPooling2D((3, 3), strides=(1, 1), padding='same')(input_img)

	# 卷积层，数据为(1000, 256,256,64)
	tower_3 = Conv2D(64, (1, 1), padding='same', activation='relu')(tower_3)
	return input_img,tower_3

def globalModel():
	#获取输入
	input_img,first = FirstConv2DModel()
	input_img,second = SecondConv2DModel()
	input_img,third = ThirdConv2DModel()

	#合并输出，数据为(1000, 256,256,192)
	output = keras.layers.concatenate([first, second, third])

	#拍扁，数据为(1000, 12582912)
	output = Flatten()(output)

	#单分类层
	output = Dense(1, activation='sigmoid', name='main_output')(output)
	return input_img,output

inputs,outputs = globalModel()
model = Model(inputs,outputs)
print(model.summary())
model.compile(optimizer='rmsprop', loss='binary_crossentropy')
model.fit(img_train, label_train, epochs=50, batch_size=32)