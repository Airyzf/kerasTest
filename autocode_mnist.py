import numpy as np
np.random.seed(1337)
from keras.datasets import mnist
from keras.models import Model
from keras.layers import Dense,Input
import matplotlib as plt

(x_train,y_train),(x_test,y_test)=mnist.load_data()

print('before...')
print(x_train.shape)
print(x_test.shape)
x_train=x_train.astype('float32')/255-0.5
x_test=x_test.astype('float32')/255-0.5
x_train=x_train.reshape((x_train.shape[0]),-1)
x_test=x_test.reshape((x_test.shape[0]),-1)
print('after...')
print(x_train.shape)
print(x_test.shape)
