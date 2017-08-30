from keras.models import Sequential
from keras.layers import LSTM, Dense
import keras as keras
import numpy as np

num_classes = 10
data_dim = 16
timesteps = 20

# 这个例子是演示使用层叠lstm来处理序列的问题

# 生成数据，1000个含有为20单词的句子，每个单词用一个不大于data_dims的数字来代表
x_train_mm = np.random.randint(data_dim, size=(1000, timesteps))
y_train = keras.utils.to_categorical(np.random.randint(10, size=(1000, 1)),num_classes=10)
x_test_mm = np.random.randint(data_dim, size=(100, timesteps))
y_test = keras.utils.to_categorical(np.random.randint(10, size=(100, 1)),num_classes=10)

# 处理为one-hot编码
x_train = np.zeros((len(x_train_mm),timesteps,data_dim))
for i in range (0,len(x_train_mm)):
	x_train[i] = keras.utils.to_categorical(x_train_mm[i],num_classes=data_dim)
x_test = np.zeros((len(x_test_mm),timesteps,data_dim))
for i in range (0,len(x_test_mm)):
	x_test[i] = keras.utils.to_categorical(x_test_mm[i],num_classes=data_dim)

model = Sequential()

# lstm层，32个特征层，输入数据为one-hot编码，由于设置为return_sequences，所以每个word都输出一次数据，
# 所以数据从(1000,20,16)转换到(1000,20,32)
model.add(LSTM(32, return_sequences=True,input_shape=(timesteps, data_dim)))

# 再次嵌套一个lstm层，数据从(1000,20,32)转换到(1000,20,32)
model.add(LSTM(32, return_sequences=True))

# 再次嵌套一个lstm层，总共层叠了三层lstm，但这一层没有设置return_sequences，
# 所以数据从(1000,20,32)转换到(1000,32)
model.add(LSTM(32))

# 多分类层
model.add(Dense(10, activation='softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

print(model.summary())

model.fit(x_train, y_train,
          batch_size=64, epochs=5)