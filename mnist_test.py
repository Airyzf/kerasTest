from keras.models import Sequential,load_model
from keras.layers import Dense, Activation
from keras.datasets import mnist
from keras.utils import np_utils

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.reshape(-1, 28 * 28)
x_test = x_test.reshape(-1, 28 * 28)
y_train = np_utils.to_categorical(y_train, 10)
y_test = np_utils.to_categorical(y_test, 10)

model = Sequential()
model.add(Dense(input_dim=784, output_dim=100))
model.add(Activation('sigmoid'))
model.add(Dense(output_dim=100))
model.add(Activation('sigmoid'))
model.add(Dense(output_dim=10))
model.add(Activation('softmax'))

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.fit(x_train, y_train, batch_size=32, epochs=10)

print("testing...")
loss, acc = model.evaluate(x_test, y_test)
print("\n loss: ", loss)
print(" acc: ", acc)

# model=load_model('minit_model.pb')
predit = model.predict(x_test[1].reshape(-1, 784))
print("predit: ", predit)
print("x_test[1]: ", predit.argmax())

model.save("minit_model.pb")

