from keras.layers import Input, Embedding, LSTM, Dense
from keras.models import Model
import numpy as np
import keras as keras

# 生成随机数据
text_train = np.random.randint(10000,size=(1000, 100))
text_label = np.random.randint(2, size=(1000, 1))
time_train = np.random.random((1000, 5))
time_label = np.random.randint(2, size=(1000, 1))

def textModel():
	# 文本模型，100个单词的句子，数据为(1000,100)
	main_input = Input(shape=(100,), dtype='int32', name='main_input')

	# Embedding层,变为(1000,100,512)
	x = Embedding(output_dim=512, input_dim=10000, input_length=100)(main_input)

	# lstm层,变为(1000,32)
	lstm_out = LSTM(32)(x)
	return main_input,lstm_out

def timeModel():
	# 时间模型，数据为(1000,5)
	auxiliary_input = Input(shape=(5,), name='aux_input')
	return auxiliary_input ,auxiliary_input

def globalModel():
	textInput,textOutput = textModel()
	timeInput,timeOutput = timeModel()

	# 文本模型的辅助输出，单分类输出
	auxiliary_output = Dense(1, activation='sigmoid', name='aux_output')(textOutput)

	# 合并文本模型的(1000,32)和(1000,5)为(1000,37)
	x = keras.layers.concatenate([textOutput, timeOutput])

	# 全连接层，数据为(1000,64)
	x = Dense(64, activation='relu')(x)

	# 全连接层，数据为(1000,64)
	x = Dense(64, activation='relu')(x)

	# 全连接层，数据为(1000,64)
	x = Dense(64, activation='relu')(x)

	# 合并的主输出，单分类输出
	main_output = Dense(1, activation='sigmoid', name='main_output')(x)

	return [textInput, timeInput], [main_output, auxiliary_output]

inputs,outputs = globalModel()
model = Model(inputs=inputs,outputs=outputs)
print(model.summary())
model.compile(optimizer='rmsprop', loss=['binary_crossentropy','binary_crossentropy'],
              loss_weights=[1., 0.2])

model.fit([text_train, time_train], [text_label, time_label],
          epochs=50, batch_size=32)