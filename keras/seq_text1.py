from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.layers import Embedding
from keras.layers import LSTM
import numpy as np

max_features = 10000
# 生成数据，1000个含有为20单词的句子，每个单词用一个不大于max_features的数字来代表
x_train =np.random.randint(max_features, size=(1000, 20))
y_train = np.random.randint(2, size=(1000, 1))
x_test =np.random.randint(max_features, size=(100, 20))
y_test = np.random.randint(2, size=(100, 1))


model = Sequential()

# 嵌入层，将句子在线向量化，数据从(1000,20)转换到(1000,20,256)
model.add(Embedding(max_features, output_dim=256))

# lstm层，提取句子的128个特征，数据从(1000,20,256)转换到了(1000,128)
model.add(LSTM(128))

# 正则化层，数据从(1000,128)转换到了(1000,128)
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