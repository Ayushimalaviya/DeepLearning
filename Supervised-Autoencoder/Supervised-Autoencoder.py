#!/usr/bin/env python
# coding: utf-8

#  Build a Supervised Autoencoder.
# PCA and the standard autoencoder are unsupervised dimensionality reduction methods, and their learned features are not discriminative. If you build a classifier upon the low-dimenional features extracted by PCA and autoencoder, you will find the classification accuracy very poor.
# Linear discriminant analysis (LDA) is a traditionally supervised dimensionality reduction method for learning low-dimensional features which are highly discriminative. Likewise, can we extend autoencoder to supervised leanring?
# ![Network Structure](https://github.com/wangshusen/CS583A-2019Spring/blob/master/homework/HM5/supervised_ae.png?raw=true "NetworkStructure")
# 

# 1. Data preparation

# 1.1. Load data
# In[1]:


from keras.datasets import mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.reshape(60000, 28*28).astype('float32') / 255.
x_test = x_test.reshape(10000, 28*28).astype('float32') / 255.

print('Shape of x_train: ' + str(x_train.shape)) 
print('Shape of x_test: ' + str(x_test.shape))
print('Shape of y_train: ' + str(y_train.shape))
print('Shape of y_test: ' + str(y_test.shape))


# 1.2. One-hot encode the labels
# 
# In the input, a label is a scalar in $\{0, 1, \cdots , 9\}$. One-hot encode transform such a scalar to a $10$-dim vector. E.g., a scalar ```y_train[j]=3``` is transformed to the vector ```y_train_vec[j]=[0, 0, 0, 1, 0, 0, 0, 0, 0, 0]```.
# 
# 1. Define a function ```to_one_hot``` that transforms an $n\times 1$ array to a $n\times 10$ matrix.
# 
# 2. Apply the function to ```y_train``` and ```y_test```.

import numpy as np

def to_one_hot(y, num_class=10):
    results = np.zeros((len(y), num_class))
    for i, label in enumerate(y):
        results[i, label] = 1.
    return results

y_train_vec = to_one_hot(y_train)
y_test_vec = to_one_hot(y_test)

print('Shape of y_train_vec: ' + str(y_train_vec.shape))
print('Shape of y_test_vec: ' + str(y_test_vec.shape))

print(y_train[0])
print(y_train_vec[0])


# 1.3. Randomly partition the training set to training and validation sets
# 
# Randomly partition the 60K training samples to 2 sets:
# * a training set containing 10K samples;
# * a validation set containing 50K samples. 

rand_indices = np.random.permutation(60000)
train_indices = rand_indices[0:10000]
valid_indices = rand_indices[10000:20000]

x_val = x_train[valid_indices, :]
y_val = y_train_vec[valid_indices, :]

x_tr = x_train[train_indices, :]
y_tr = y_train_vec[train_indices, :]

print('Shape of x_tr: ' + str(x_tr.shape))
print('Shape of y_tr: ' + str(y_tr.shape))
print('Shape of x_val: ' + str(x_val.shape))
print('Shape of y_val: ' + str(y_val.shape))


# 2. Build an unsupervised  autoencoder and tune its hyper-parameters

# 2.1. Build the model

from keras.layers import Input, Dense
from keras import models

input_img = Input(shape=(784,), name='input_img')

encode1 = Dense(256, activation = 'relu')(input_img)
encode2 = Dense(64, activation = 'relu')(encode1)
encode3 = Dense(64, activation = 'relu')(encode2)
# <Add more layers...>
# <the output of encoder network>
bottleneck = Dense(2, activation = 'relu')(encode3)
# <add a dense layer taking bottleneck as input>
decode1 =  Dense(32, activation = 'relu')(bottleneck)
decode2 =  Dense(128, activation = 'relu')(decode1)
decode3 =  Dense(64, activation = 'relu')(decode2)
decode4 = Dense(784, activation = 'sigmoid')(decode3)
ae = models.Model(input_img, decode4)
ae.summary()

# print the network structure to a PDF file

from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot, plot_model

SVG(model_to_dot(ae, show_shapes=False).create(prog='dot', format='svg'))

plot_model(
    model=ae, show_shapes=False,
    to_file='unsupervised_ae.pdf'
)

# you can find the file "unsupervised_ae.pdf" in the current directory.


# 2.2. Train the model and tune the hyper-parameters
from tensorflow.keras import optimizers
learning_rate = 0.003 # to be tuned!
ae.compile(loss='mean_squared_error',
           optimizer=optimizers.Adam(learning_rate=learning_rate))
history = ae.fit(x_tr, x_tr, 
                 batch_size=128, 
                 epochs=100, 
                 validation_data=(x_val, x_val))

import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(len(loss))

plt.plot(epochs, loss, 'bo', label='Training Loss')
plt.plot(epochs, val_loss, 'r', label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()


# 2.3. Visualize the reconstructed test images
ae_output = ae.predict(x_test).reshape((10000, 28, 28))

ROW = 5
COLUMN = 4

x = ae_output
fname = 'reconstruct_ae.pdf'

fig, axes = plt.subplots(nrows=ROW, ncols=COLUMN, figsize=(8, 10))
for ax, i in zip(axes.flat, np.arange(ROW*COLUMN)):
    image = x[i].reshape(28, 28)
    ax.imshow(image, cmap='gray')
    ax.axis('off')

plt.tight_layout()
plt.savefig(fname)
plt.show()

# 2.4. Evaluate the model on the test set
loss = ae.evaluate(x_test, x_test)
print('loss = ' + str(loss))


# 2.5. Visualize the low-dimensional features
# build the encoder network
ae_encoder = models.Model(input_img, bottleneck)
ae_encoder.summary()

# extract low-dimensional features from the test data
encoded_test = ae_encoder.predict(x_test)
print('Shape of encoded_test: ' + str(encoded_test.shape))

colors = np.array(['r', 'g', 'b', 'm', 'c', 'k', 'y', 'purple', 'darkred', 'navy'])
colors_test = colors[y_test]

import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

fig = plt.figure(figsize=(8, 8))
plt.scatter(encoded_test[:, 0], encoded_test[:, 1], s=10, c=colors_test, edgecolors=colors_test)
plt.axis('off')
plt.tight_layout()
fname = 'ae_code.pdf'
plt.savefig(fname)
 
# Judging from the visualization, the low-dim features seems not discriminative, as 2D features from different classes are mixed. Let quantatively find out whether they are discriminative.
# extract the 2D features from the training, validation, and test samples
f_tr = ae_encoder.predict(x_tr)
f_val = ae_encoder.predict(x_val)
f_te = ae_encoder.predict(x_test)

print('Shape of f_tr: ' + str(f_tr.shape))
print('Shape of f_te: ' + str(f_te.shape))

from keras.layers import Dense, Input
from keras import models

input_feat = Input(shape=(2,))

hidden1 = Dense(128, activation='relu')(input_feat)
hidden2 = Dense(128, activation='relu')(hidden1)
output = Dense(10, activation='softmax')(hidden2)

classifier = models.Model(input_feat, output)
classifier.summary()
classifier.compile(loss='categorical_crossentropy',
                  optimizer=optimizers.RMSprop(learning_rate=1E-4),
                  metrics=['acc'])
history = classifier.fit(f_tr, y_tr, batch_size=32, epochs=30, validation_data=(f_val, y_val))


# Build a supervised autoencode model for learning low-dimensional discriminative features.
#  4. Build a supervised autoencoder model
# ![Network Structure](https://github.com/wangshusen/CS583A-2019Spring/blob/master/homework/HM5/supervised_ae.png?raw=true "NetworkStructure")
# 4.1. Build the network
# build the supervised autoencoder network
from keras.layers import Input, Dense, BatchNormalization, Dropout, Activation
from keras import models

input_img = Input(shape=(784,), name='input_img')

encode1 = Dense(256, name='encode_1')(input_img)
encode1 = Dropout(0.5)(encode1)
encode1 = BatchNormalization()(encode1)
encode1 = Activation('relu')(encode1)

encode2 = Dense(64, name='encode_2')(encode1)
encode2 = Dropout(0.5)(encode2)
encode2 = BatchNormalization()(encode2)
encode2 = Activation('relu')(encode2)

encode3 = Dense(64, name='encode_3')(encode2)
# encode3 = Dropout(0.5)(encode3)
encode3 = BatchNormalization()(encode3)
encode3 = Activation('relu')(encode3)

# The width of the bottleneck layer must be exactly 2.
bottleneck = Dense(2, activation='relu', name='Bottlencek')(encode3)
bottleneck = BatchNormalization()(bottleneck)
bottleneck = Activation('relu')(bottleneck)

# decoder network
decode1 = Dense(32, name='decode_1')(bottleneck)
# decode1 = Dropout(0.5)(decode1)
decode1 = BatchNormalization()(decode1)
decode1 = Activation('relu')(decode1)

decode2 = Dense(128, name='decode_2')(decode1)
# decode2 = Dropout(0.5)(decode2)
decode2 = BatchNormalization()(decode2)
decode2 = Activation('relu')(decode2)

decode3 = Dense(64, name='decode_3')(decode2)
decode3 = Dropout(0.5)(decode3)
decode3 = BatchNormalization()(decode3)
decode3 = Activation('relu')(decode3)

decode4 = Dense(784, name='decode_4')(decode3)
decode4 = Dropout(0.5)(decode4)
decode4 = BatchNormalization()(decode4)
decode4 = Activation('relu')(decode4)

# build a classifier upon the bottleneck layer
classifier1 = Dense(64, name='classifier_1')(bottleneck)
classifier1 = Dropout(0.5)(classifier1)
classifier1 = BatchNormalization()(classifier1)
classifier1 = Activation('relu')(classifier1)

classifier2 = Dense(64, name='classifier_2')(classifier1)
# classifier2 = Dropout(0.2)(classifier2)
classifier2 = BatchNormalization()(classifier2)
classifier2 = Activation('relu')(classifier2)
classifier3 = Dense(10, name='classifier_3', activation='softmax')(classifier2)


# In[42]:


# connect the input and the two outputs
sae = models.Model(input_img, [decode4, classifier3])
sae.summary()
# print the network structure to a PDF file

from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot, plot_model

SVG(model_to_dot(sae, show_shapes=False).create(prog='dot', format='svg'))

plot_model(
    model=sae, show_shapes=False,
    to_file='supervised_ae.pdf'
)


# 4.2. Train the new model and tune the hyper-parameters
from tensorflow.keras import optimizers

sae.compile(loss=['mean_squared_error', 'categorical_crossentropy'],
            loss_weights=[0.8, 0.65], # to be tuned
            optimizer=optimizers.RMSprop(learning_rate=1E-3))

history = sae.fit(x_tr, [x_tr, y_tr], batch_size=32, epochs=100, validation_data=(x_val, [x_val, y_val]))

import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(len(loss))

plt.plot(epochs, loss, 'bo', label='Training Loss')
plt.plot(epochs, val_loss, 'r', label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()


# 4.3. Visualize the reconstructed test images

sae_output = sae.predict(x_test)[0].reshape((10000, 28, 28))

ROW = 5
COLUMN = 4

x = sae_output
fname = 'reconstruct_sae.pdf'

fig, axes = plt.subplots(nrows=ROW, ncols=COLUMN, figsize=(8, 10))
for ax, i in zip(axes.flat, np.arange(ROW*COLUMN)):
    image = x[i].reshape(28, 28)
    ax.imshow(image, cmap='gray')
    ax.axis('off')

plt.tight_layout()
plt.savefig(fname)
plt.show()


# 4.4. Visualize the low-dimensional features
# 
# 
# build the encoder model
sae_encoder = models.Model(input_img, bottleneck)
sae_encoder.summary()

# extract test features
encoded_test = sae_encoder.predict(x_test)
print('Shape of encoded_test: ' + str(encoded_test.shape))

colors = np.array(['r', 'g', 'b', 'm', 'c', 'k', 'y', 'purple', 'darkred', 'navy'])
colors_test = colors[y_test]

import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

fig = plt.figure(figsize=(8, 8))
plt.scatter(encoded_test[:, 0], encoded_test[:, 1], s=10, c=colors_test, edgecolors=colors_test)
plt.axis('off')
plt.tight_layout()
fname = 'sae_code.pdf'
plt.savefig(fname)

# extract 2D features from the training, validation, and test samples
f_tr = sae_encoder.predict(x_tr)
f_val = sae_encoder.predict(x_val)
f_te = sae_encoder.predict(x_test)

# build a classifier which takes the 2D features as input
from keras.layers import *
from keras import models

input_feat = Input(shape=(2,))

# build the classifier network
dense1 = Dense(128, activation='relu')(input_feat)
dense2 = Dense(64, activation='relu')(dense1)
dense3 = Dense(32, activation='relu')(dense2)
output = Dense(10, activation='sigmoid')(dense3)

# define the classifier model
classifier = models.Model(input_feat, output)
classifier.summary()
classifier.compile(loss='categorical_crossentropy',
                  optimizer=optimizers.RMSprop(learning_rate=1E-4),
                  metrics=['acc'])
history = classifier.fit(f_tr, y_tr, batch_size=32, epochs=30, validation_data=(f_val, y_val))

# evaluate your model on the never-seen-before test data
# write your code here:
lossAccuracy = classifier.evaluate(f_te, y_test_vec)

print('Accuracy:', lossAccuracy[1] * 100)

