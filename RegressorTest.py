# -*- coding: utf-8 -*-
"""
Created on Thu Apr 13 23:49:34 2017

@author: yzf_f
"""

import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Dense
import matplotlib.pyplot as plt

np.random.seed(1337)
X=np.linspace(-1,1,200)
np.random.shuffle(X)
Y=0.5*X+2+np.random.normal(0,0.05,(200,))
plt.scatter(X,Y)
plt.show()

X_train,Y_train=X[:160],Y[:160]
X_test,Y_test=X[160:],Y[160:]

model=Sequential()
model.add(Dense(output_dim=1,input_dim=1))

model.compile(loss='mse',optimizer='sgd')

print('Training......')
for step in range(301):
    cost=model.train_on_batch(X_train,Y_train)
    if step%100==0:
        print('train cost: ',cost)
        
print("\nTesting")
cost=model.evaluate(X_test,Y_test,batch_size=40)
print('test cost:',cost)
W,b=model.layers[0].get_weights()
print('Weights',W,'\nbiases',b)

Y_pred=model.predict(X_test)
plt.scatter(X_test,Y_test)
plt.plot(X_test,Y_pred)
plt.show()
