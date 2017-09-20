# 关闭证书验证
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

from model import *
from dense import *
from activation import *
from loss import *
from optimizer import *
from util import *
from dropout import *
import numpy as np
from keras.datasets import mnist

batch_size = 128
num_classes = 10
epochs = 2

# 训练数据
(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.reshape(60000, 784)
x_test = x_test.reshape(10000, 784)
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255

y_train = y_train.reshape(60000,1)
y_test = y_test.reshape(10000,1)
y_train = to_categorical(y_train, num_classes)
y_test = to_categorical(y_test, num_classes)

# 训练

model = Model()
model.add(Dense(512,input_shape=(784,)))
model.add(Activation("relu"))
model.add(Dropout(0.25))
model.add(Dense(512))
model.add(Activation("relu"))
model.add(Dropout(0.25))
model.add(Dense(10))

optimizer = Optimizer(rate=0.01,momentum=0.9)
loss = Loss("softmax_categorical_crossentropy")
model.compile(optimizer=optimizer,loss=loss)

model.fit(x_train,y_train,epochs=epochs,batch_size=batch_size,verbose=True)

loss,acc = model.evaluate(x_test,y_test)
print(loss,acc)