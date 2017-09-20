# 关闭证书验证
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

# 加入上级目录
import path, sys 
import folder = path.path(__file__).abspath() 
sys.path.append(folder.parent) 

from bp.model import *
from bp.dense import *
from bp.activation import *
from bp.loss import *
from bp.optimizer import *
from bp.util import *
from conv import *
from pooling import *
from flatten import *
import numpy as np
from keras.datasets import mnist

batch_size = 128
num_classes = 10
epochs = 2

# 训练数据
(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.reshape(60000, 1,28,28)
x_test = x_test.reshape(10000, 1,28,28)
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
model.add(Conv2D(32,kernal_size=(3,3),input_shape=(1,28,28)))
model.add(Activation("relu"))
model.add(Conv2D(64,kernal_size=(3,3))
model.add(Activation("relu"))
model.add(MaxPooling2D(64,pool_size=(2,2))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128))
model.add(Activation("relu"))
model.add(Dropout(0.5))
model.add(Dense(num_classes))

optimizer = Optimizer(rate=0.01,momentum=0.9)
loss = Loss("softmax_categorical_crossentropy")
model.compile(optimizer=optimizer,loss=loss)

model.fit(x_train,y_train,epochs=epochs,batch_size=batch_size,verbose=True)

loss,acc = model.evaluate(x_test,y_test)
print(loss,acc)