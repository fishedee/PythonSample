from keras.models import Sequential
from keras.layers import LSTM, Dense
import keras as keras
import numpy as np

data_dim = 16
timesteps = 8
num_classes = 10
batch_size = 32

# 这个例子是演示使用层叠lstm来处理超长序列的问题，超长序列被分割为多个句子
# 送入到lstm中，其中batch_size指定了多少个batch的句子才为一个完整序列。

# 生成数据，1000个含有为20单词的句子，每个单词用一个不大于data_dims的数字来代表
x_train_mm = np.random.randint(data_dim, size=(batch_size*10, timesteps))
y_train = keras.utils.to_categorical(np.random.randint(10, size=(batch_size*10, 1)),num_classes=10)
x_test_mm = np.random.randint(data_dim, size=(batch_size*3, timesteps))
y_test = keras.utils.to_categorical(np.random.randint(10, size=(batch_size*3, 1)),num_classes=10)

# 处理为one-hot编码
x_train = np.zeros((len(x_train_mm),timesteps,data_dim))
for i in range (0,len(x_train_mm)):
	x_train[i] = keras.utils.to_categorical(x_train_mm[i],num_classes=data_dim)
x_test = np.zeros((len(x_test_mm),timesteps,data_dim))
for i in range (0,len(x_test_mm)):
	x_test[i] = keras.utils.to_categorical(x_test_mm[i],num_classes=data_dim)

model = Sequential()

# 含状态的lstm层，注意stateful为True，而且指定了batch_size，说明连续batch_size个输入为一个序列，它们之间的状态是共享的
# 数据从(batch_size*10,8,16)转换为(batch_size*10,8,32)
model.add(LSTM(32, return_sequences=True, stateful=True,batch_input_shape=(batch_size, timesteps, data_dim)))

# 依旧为含状态的lstm层，数据从(batch_size*10,8,32)转换为(batch_size*10,8,32)
model.add(LSTM(32, return_sequences=True, stateful=True))

# 依旧为含状态的lstm层，但没有return_sequences,数据从(batch_size*10,8,32)转换为(batch_size*10,32)
model.add(LSTM(32, stateful=True))

# 全分类层
model.add(Dense(10, activation='softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

print(model.summary())

model.fit(x_train, y_train,
          batch_size=batch_size, epochs=5, shuffle=False,
          validation_data=(x_test, y_test))