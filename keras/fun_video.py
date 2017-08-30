from keras.layers import Input, Embedding, LSTM, Dense,Conv2D,MaxPooling2D,Flatten,TimeDistributed
from keras.models import Model,Sequential
import numpy as np
import keras as keras

# 生成随机数据，一堆图片，以及label标签
video_train = np.random.random((10, 100,224, 224,3))
text_train = np.random.randint(10000,size=(10, 100))
label_train = keras.utils.to_categorical(np.random.randint(1000, size=(10, 1)), num_classes=1000)

def getVisualModel():
	#输入为(100,224,224,3)，输出为(100,160000)
	vision_model = Sequential()
	vision_model.add(Conv2D(64, (3, 3),activation='relu', padding='same', input_shape=( 224, 224,3)))
	vision_model.add(Conv2D(64, (3, 3), activation='relu'))
	vision_model.add(MaxPooling2D((2, 2)))
	vision_model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
	vision_model.add(Conv2D(128, (3, 3), activation='relu'))
	vision_model.add(MaxPooling2D((2, 2)))
	vision_model.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
	vision_model.add(Conv2D(256, (3, 3), activation='relu'))
	vision_model.add(Conv2D(256, (3, 3), activation='relu'))
	vision_model.add(MaxPooling2D((2, 2)))
	vision_model.add(Flatten())
	return vision_model

def getVideoModel():
	# 输入为(10,100, 224, 224,3)
	video_input = Input(shape=(100,  224, 224,3))
	vision_model = getVisualModel()
	# 每帧执行VisualModel特征提取，数据为(10,100,160000)
	encoded_frame_sequence = TimeDistributed(vision_model)(video_input)

	#输出为(10,256)
	encoded_video = LSTM(256)(encoded_frame_sequence)
	return video_input,encoded_video

def getTextModel():
	#输入为(10,100)，输出为(10,256)
	question_input = Input(shape=(100,), dtype='int32')
	embedded_question = Embedding(input_dim=10000, output_dim=256, input_length=100)(question_input)
	encoded_question = LSTM(256)(embedded_question)
	return Model(question_input,encoded_question)

def getQuestionModel():
	#输入为(10,100)，输出为(10,256)
	video_question_input = Input(shape=(100,), dtype='int32')
	encoded_video_question = getTextModel()(video_question_input)
	return video_question_input,encoded_video_question

def globalModel():
	videoInput,videoOutput = getVideoModel()
	questionInput,questionOutput = getQuestionModel()
	
	#合并层，数据为(10,512)
	merged = keras.layers.concatenate([videoOutput, questionOutput])

	#全连接层，多分类
	output = Dense(1000, activation='softmax')(merged)
	return [videoInput,questionInput],[output]

inputs,outputs = globalModel()
model = Model(inputs=inputs,outputs=outputs)
print(model.summary())
model.compile(optimizer='rmsprop', loss='categorical_crossentropy')
model.fit([video_train,text_train], [label_train], epochs=50, batch_size=32)