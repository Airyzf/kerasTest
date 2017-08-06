import numpy as np

np.random.seed(1337)
from keras.datasets import cifar10
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Activation, Convolution2D, MaxPool2D, Flatten, Dropout
from keras.optimizers import Adam

(X_train, y_train), (X_test, y_test) = cifar10.load_data()
X_train = X_train.reshape(-1, 3, 32, 32) / 255
X_test = X_test.reshape(-1, 3, 32, 32) / 255
y_test = np_utils.to_categorical(y_test, num_classes=10)
y_train = np_utils.to_categorical(y_train, num_classes=10)

model = Sequential()
model.add(Convolution2D(
    filters=32, kernel_size=(5, 5), padding='same', input_shape=(3, 32, 32)
))
model.add(Activation('relu'))
model.add(MaxPool2D(
    pool_size=(2, 2), padding='same'
))
model.add(Dropout(0.25))

model.add(Convolution2D(
    filters=64, kernel_size=(5, 5), padding='same'
))
model.add(Activation('relu'))
model.add(MaxPool2D(
    strides=(2, 2), padding='same'
))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(640))
model.add(Activation('relu'))
model.add(Dense(10))
model.add(Dropout(0.25))
model.add(Activation('softmax'))
adam = Adam()

model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy'])

print("Train...")
accuracy = 0
for n in range(10):
    if accuracy < 80:
        model.fit(X_train, y_train, epochs=1, batch_size=32, )
        loss, accuracy = model.evaluate(X_test, y_test)
    else:
        print('accuracy 已经大于80')


# print("Test...")
# loss, accuracy = model.evaluate(X_test, y_test)
# print("loss : ", loss)
# print("accuracy : ", accuracy)
