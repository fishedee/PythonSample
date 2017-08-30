import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.optimizers import SGD

# 生成数据，100张图，每张图的100x100，深度为3的图片，由于我们使用的是Tensorflow引擎，所以channel在最后一个维度
x_train = np.random.random((100, 100, 100, 3))
y_train = keras.utils.to_categorical(np.random.randint(10, size=(100, 1)), num_classes=10)
x_test = np.random.random((20, 100, 100, 3))
y_test = keras.utils.to_categorical(np.random.randint(10, size=(20, 1)), num_classes=10)

model = Sequential()
# Conv2D层，32个特征层，3x3的卷积核，激活函数为relu
# 经过这一步以后，数据从(100,100,100,3)变换到了(100,100,100,32)
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(100, 100, 3)))

# Conv2D层，32个特征层，3x3的卷积核，激活函数为relu
# 再次Conv2d，是数据的再卷积，数据从(100,100,100,32)变换到了(100,100,100,32)
model.add(Conv2D(32, (3, 3), activation='relu'))

# MaxPooling2D层，是2d的取最大值的池化层，池化核为2x2，表示2x2的数据会合并到1个数据中
# 经过这一步以后，数据从(100,100,100,32)变换到了(100,50,50,32)
model.add(MaxPooling2D(pool_size=(2, 2)))

# 加入正则化层，数据从(100,50,50,32)变换到了(100,50,50,32)
model.add(Dropout(0.25))

# 依然是Conv2D层，数据从(100,50,50,32)变换到了(100,50,50,64)
model.add(Conv2D(64, (3, 3), activation='relu'))

# 依然是Conv2D层，数据从(100,50,50,64)变换到了(100,50,50,64)
model.add(Conv2D(64, (3, 3), activation='relu'))

# 依然是MaxPooling2D层，数据从(100,50,50,64)变换到了(100,25,25,64)
model.add(MaxPooling2D(pool_size=(2, 2)))

# 依然是正则化层，数据从(100,25,25,64)变换到了(100,25,25,64)
model.add(Dropout(0.25))

# 这个层很特别，直接拍扁了多维度数据转换成一层，数据从(100,25,25,64)变换到了(100,40000)
model.add(Flatten())

# 全连接层，数据从(100,40000)变换到了(100,256)
model.add(Dense(256, activation='relu'))

# 依然是正则化层，数据从(100,256)变换到了(100,256)
model.add(Dropout(0.5))

# 最后一层分类层，分类为10个类别
model.add(Dense(10, activation='softmax'))

sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd)

print(model.summary())

model.fit(x_train, y_train, batch_size=32, epochs=10)
score = model.evaluate(x_test, y_test, batch_size=32)

print(score)
