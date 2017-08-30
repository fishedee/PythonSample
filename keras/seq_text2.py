from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.layers import Embedding
from keras.layers import Conv1D, GlobalAveragePooling1D, MaxPooling1D
import numpy as np
import keras as keras

max_features = 100
seq_length = 20
# 生成数据，1000个含有为20单词的句子，每个单词用一个不大于max_features的数字来代表
x_train_mm = np.random.randint(max_features, size=(1000, seq_length))
y_train = np.random.randint(2, size=(1000, 1))
x_test_mm = np.random.randint(max_features, size=(100, seq_length))
y_test = np.random.randint(2, size=(100, 1))

# 注意跟lstm不一样，放入cnn训练的文本数据必须要是one-hot编码
x_train = np.zeros((len(x_train_mm),seq_length,max_features))
for i in range (0,len(x_train_mm)):
	x_train[i] = keras.utils.to_categorical(x_train_mm[i],num_classes=max_features)
x_test = np.zeros((len(x_test_mm),seq_length,max_features))
for i in range (0,len(x_test_mm)):
	x_test[i] = keras.utils.to_categorical(x_test_mm[i],num_classes=max_features)

model = Sequential()

# 卷积层，将句子中每个单词的邻近的三个单词提取出来作64个特征，数据从(1000,20,100)转换到了(1000,20,64)
model.add(Conv1D(64, 3, activation='relu', input_shape=(seq_length, max_features)))

# 卷积层，数据从(1000,20,64)转换到了(1000,20,64)
model.add(Conv1D(64, 3, activation='relu'))

# 池化层，数据从(1000,20,64)转换到(1000,7,64)
model.add(MaxPooling1D(3))

# 卷积层，数据从(1000,7,64)转换到(1000,7,128)
model.add(Conv1D(128, 3, activation='relu'))

# 卷积层，数据从(1000,7,128)转换到(1000,7,128)
model.add(Conv1D(128, 3, activation='relu'))

# 全局池化层，相当于在池化层的基础上将池化核设置为全局，然后取单个平均值，也就是说，数据从(1000,7,128)转换到(1000,128)
model.add(GlobalAveragePooling1D())

# 正则化层，数据从(1000,128)转换到(1000,128)
model.add(Dropout(0.5))

# 单分类层
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

print(model.summary())

model.fit(x_train, y_train, batch_size=16, epochs=10)
score = model.evaluate(x_test, y_test, batch_size=16)

print(score)