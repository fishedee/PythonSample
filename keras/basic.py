from keras.models import Sequential
from keras.layers import Dense, Activation
import numpy as np

#加入训练数据
x_train = np.random.random((1000, 100))
y_train = np.random.randint(2, size=(1000, 1))

#建立单向图，也就是序列模型
model = Sequential()

#全连接层,input_dim是输入层的维度
model.add(Dense(units=64, input_dim=100))
model.add(Activation("relu"))

#全连接层
model.add(Dense(units=1))
model.add(Activation("sigmoid"))

#编译模型，指定损失函数，训练方法和性能指标
model.compile(loss='binary_crossentropy', optimizer='sgd', metrics=['accuracy'])

#训练数据
model.fit(x_train, y_train, epochs=100, batch_size=32,verbose=1)

#测试数据
loss_and_metrics = model.evaluate(x_train, y_train, batch_size=128)
print(loss_and_metrics)