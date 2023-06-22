#!/usr/bin/env python
# coding: utf-8

from keras.datasets import mnist
from keras import models, layers, optimizers
import pandas as pd
import numpy as np

(x_train, y_train),(x_test, y_test) = mnist.load_data()
print('X train shape:' + str(x_train.shape))
print('Y train shape:' + str(y_train.shape))
print('X test shape:' + str(x_test.shape))
print('Y test shape:' + str(y_test.shape))

x_train_vec = x_train.reshape(60000, 784)
x_test_vec = x_test.reshape(10000, 784)
print('shape of x_train_vec is:'+ str(x_train_vec.shape))

#one hot encoding of y dataset values
def to_one_hot_encode(y, dimensions = 10):
  labels_list = list(y)
  #array = np.zeros((len(y), dimensions))
  results_list = list(map(lambda y: [1 if i == y else 0 for i in range(10)], labels_list))
  results = np.array(results_list)
  return results

y_train_vec = to_one_hot_encode(y_train)
y_test_vec = to_one_hot_encode(y_test)
print('shape of y test vector:' +str(y_train_vec.shape))
print('shape of y train vector:' +str(y_test_vec.shape))

print('shape of x_train_vec is:'+ str(x_train_vec.shape))
print('shape of x_train_vec is:'+ str(x_test_vec.shape))
print('shape of y test vector:' +str(y_train_vec.shape))
print('shape of y train vector:' +str(y_test_vec.shape))

#n_indices = x_train_vec.shape[0]
rand_indices = np.random.permutation(60000)
train_indices = rand_indices[0:50000]
valid_indices = rand_indices[50000:60000]
x_valid_vec = x_train_vec[valid_indices, :]
y_valid_vec = y_train_vec[valid_indices, :]

x_train_vec = x_train_vec[train_indices, :]
y_train_vec = y_train_vec[train_indices, :] 

print('shape of x train:' +str(x_train_vec.shape))
print('shape of y train:' +str(y_train_vec.shape))
print('shape of x validate:' +str(x_valid_vec.shape))
print('shape of y validate:' +str(y_valid_vec.shape))

model = models.Sequential()
model.add(layers.Dense(10, activation='softmax', input_shape=(784,)))
model.compile(optimizers.RMSprop(learning_rate=0.01), loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

history = model.fit(x_train_vec, y_train_vec, batch_size=128, epochs=50, validation_data=(x_valid_vec, y_valid_vec))

loss_and_acc = model.evaluate(x_test_vec, y_test_vec)
print('loss= '+str(loss_and_acc[0]))
print('accuracy= '+str(round(loss_and_acc[1],2)))

d1=500
d2 = 500
model = models.Sequential()
model.add(layers.Dense(d1, activation='relu', input_shape =(784,)))
model.add(layers.Dense(d2, activation='relu'))
model.add(layers.Dense(10, activation='softmax'))
model.summary()

model.compile(optimizers.RMSprop(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

history_nn = model.fit(x_train_vec, y_train_vec, batch_size=128, epochs=50, validation_data=(x_valid_vec, y_valid_vec)) 

loss_and_acc_nn = model.evaluate(x_test_vec, y_test_vec)
print('loss =' +str(loss_and_acc_nn[0]))
print('accuracy =' +str(round(loss_and_acc_nn[1],2)))

