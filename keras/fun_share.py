from keras.layers import Input, Embedding, LSTM, Dense
from keras.models import Model
import numpy as np
import keras as keras

# 生成随机数据，两种文本，以及它们随机的相似度
text1_train = np.random.randint(10000,size=(1000, 200))
text2_train = np.random.randint(10000,size=(1000, 300))
label_train = np.random.randint(2, size=(1000, 1))

wordModel = None
def shareWordModel():
	global wordModel
	if wordModel is not None:
		return wordModel
	#可重用的文本分析模块，注意这里的shape没啥用
	main_input = Input(shape=(100,))
	embedding = Embedding(output_dim=256, input_dim=10000)(main_input)
	lstm = LSTM(32)(embedding)
	wordModel = Model(main_input,lstm)
	return wordModel

def leftWordModel():
	#左边的文本，使用200长度的文本，数据为(1000,32)
	a = Input(shape=(200,))
	wordModel = shareWordModel()
	output = wordModel(a)
	return a,output

def rightWordModel():
	#右边的文本，使用300长度的文本，数据为(1000,32)
	b = Input(shape=(300,))
	wordModel = shareWordModel()
	output = wordModel(b)
	return b,output

def globalModel():
	leftInput,leftOutput = leftWordModel()
	rightInput,rightOutput = rightWordModel()
	#合并后，数据为(1000,64)
	concatenated = keras.layers.concatenate([leftOutput, rightOutput])
	#全连接层，单分类
	out = Dense(1, activation='sigmoid')(concatenated)
	return [leftInput,rightInput],[out]

inputs,outputs = globalModel()
model = Model(inputs=inputs,outputs=outputs)
print(model.summary())
model.compile(optimizer='rmsprop', loss='binary_crossentropy')

model.fit([text1_train, text2_train], [label_train],
          epochs=50, batch_size=32)


