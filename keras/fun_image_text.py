from keras.layers import Input, Embedding, LSTM, Dense,Conv2D,MaxPooling2D,Flatten
from keras.models import Model,Sequential
import numpy as np
import keras as keras

# 生成随机数据，一堆图片，以及label标签
img_train = np.random.random((1000, 224, 224,3))
text_train = np.random.randint(10000,size=(1000, 100))
label_train = keras.utils.to_categorical(np.random.randint(1000, size=(1000, 1)), num_classes=1000)


def getVisualModel():
	#输入为(1000,224,224,3)，输出为(1000,160000)
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

def getImageModel():
	image_input = Input(shape=( 224, 224,3))
	return image_input,getVisualModel()(image_input)

def getTextModel():
	#输入为(1000,100)，输出为(1000,256)
	question_input = Input(shape=(100,), dtype='int32')
	embedded_question = Embedding(input_dim=10000, output_dim=256, input_length=100)(question_input)
	encoded_question = LSTM(256)(embedded_question)
	return question_input,encoded_question

def globalModel():
	imageInput,imageOutput = getImageModel()
	textInput,textOutput = getTextModel()
	#合并层，数据为(1000,160256)
	merged = keras.layers.concatenate([imageOutput, textOutput])

	#全连接层，多分类
	output = Dense(1000, activation='softmax')(merged)
	return [imageInput,textInput],[output]

inputs,outputs = globalModel()
model = Model(inputs=inputs,outputs=outputs)
print(model.summary())
model.compile(optimizer='rmsprop', loss='categorical_crossentropy')
model.fit([img_train,text_train], [label_train], epochs=50, batch_size=32)