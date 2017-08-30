from keras.layers import Input, Embedding, LSTM, Dense,Conv2D,MaxPooling2D,Flatten
from keras.models import Model
import numpy as np
import keras as keras

# 生成随机数据，一堆图片，以及label标签
img_train = np.random.random((1000, 256, 256,3))
label_train = np.random.randint(2, size=(1000, 1))

def globalModel():
	# 输入数据，数据为(1000,256,256,3)
	x = Input(shape=(256, 256,3))

	# 卷积层，数据为(1000,256,256,3)
	y = Conv2D(3, (3, 3), padding='same')(x)

	# 叠加层，叠加的方法为同维相加，数据为(1000,256,256,3)
	z = keras.layers.add([x, y])

	# 拍扁，数据为(1000,196608)
	z = Flatten()(z)

	#单分类层
	output = Dense(1, activation='sigmoid', name='main_output')(z)
	return x,output

inputs,outputs = globalModel()
model = Model(inputs,outputs)
print(model.summary())
model.compile(optimizer='rmsprop', loss='binary_crossentropy')
model.fit(img_train, label_train, epochs=50, batch_size=32)

