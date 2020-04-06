"""
Example of a simple neural network for MNIST classification 
using the layers API in tensorflow 2.1 
Author: Sebastian Gottwald
Project: https://github.com/sgttwld/tensorflow-2.0-simple-examples
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras import Model, Input
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

# load mnist
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0   # np.shape(x_train) = (60000,28,28) 

# define model
X = Input(shape=(28,28))
Xf = Flatten()(X)
H = Dense(200, activation='sigmoid')(Xf)
Y = Dense(10, activation='softmax')(H)
nn = Model(inputs=X, outputs=Y)

## alternatively: subclassing Model
# class create_nn(Model):
#   def __init__(self):
#     super(create_nn, self).__init__()
#     self.flatten = Flatten()
#     self.d1 = Dense(200, activation='sigmoid')
#     self.d2 = Dense(10, activation='softmax')
#   def call(self, x):
#     x = self.flatten(x)
#     x = self.d1(x)
#     return self.d2(x)
# nn = create_nn()

# add optimizer, loss, and metric
nn.compile( optimizer='adam', 
            loss=tf.keras.losses.SparseCategoricalCrossentropy(), 
            metrics=['accuracy'])

# train
nn.fit(x_train,y_train, epochs=5)
