"""
Example of a simple neural network for MNIST classification  
using the models.Sequential API in tensorflow 2.1
Author: Sebastian Gottwald
Project: https://github.com/sgttwld/tensorflow-2.0-simple-examples
"""

import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

# load mnist
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0   # np.shape(x_train) = (60000,28,28) 

# define model
nn = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(input_shape=(28, 28)),
  tf.keras.layers.Dense(200, activation='sigmoid'),
  tf.keras.layers.Dense(10, activation='softmax')
])

# add optimizer, loss, and metric
nn.compile( optimizer='adam', 
            loss=tf.keras.losses.SparseCategoricalCrossentropy(), 
            metrics=['accuracy'])

# train
nn.fit(x_train,y_train, epochs=5)
