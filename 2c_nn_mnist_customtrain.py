"""
Example of a simple neural network for MNIST classification 
using the layers API and custom training methods in tensorflow 2.1 
Author: Sebastian Gottwald
Project: https://github.com/sgttwld/tensorflow-2.0-simple-examples
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras import Model, Input
import os, math, sys, time
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 

## custom progress bar
def print_progress(i, tot, acc, acc_str, bar_length=30, wait=False):
    filled_length = int(round(bar_length * i / tot))
    # bar = '█' * filled_length + '-' * (bar_length - filled_length)
    bar = '|' * filled_length + '-' * (bar_length - filled_length)
    sys.stdout.write('\r%s/%s |%s| %s %s' % (i, tot, bar, acc_str+':', acc)),
    if i == tot-1 and not(wait):
        sys.stdout.write('\n')
    sys.stdout.flush()

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

# loss and optimizer
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy()
optimizer = tf.keras.optimizers.Adam()

# train function
@tf.function
def train(model,x,y):
    loss = lambda: loss_fn(y,model(x))
    optimizer.minimize(loss, var_list=model.trainable_variables)

# alternatively: train function with explicit gradients
# @tf.function 
# def train(model, x, y):
#     with tf.GradientTape() as g:
#         predictions = model(x, training=True)
#         loss = loss_fn(y, predictions)
#     gradients = g.gradient(loss, model.trainable_variables)
#     optimizer.apply_gradients(zip(gradients, model.trainable_variables))

# evaluation
@tf.function
def percent_corr(model,x,y):
    return tf.reduce_mean(tf.cast(tf.math.equal(tf.argmax(model(x),axis=1), tf.cast(y,tf.int64)),tf.float32))

bs = 32
numBatches = math.floor(len(x_train)/bs)

for epoch in range(5):
    t0, acc = time.time(), 0
    for b in range(numBatches):
        batch_X, batch_Y = x_train[b*bs:(b+1)*bs], y_train[b*bs:(b+1)*bs]
        train(nn, batch_X, batch_Y)
        # evaluation inside each episode
        acc = (b * acc + percent_corr(nn, batch_X,batch_Y).numpy())/(b+1)
        print_progress(b, numBatches, round(acc,4), acc_str='acc', wait=True)
    # evaluation over episodes
    T = round(time.time()-t0,2)
    acc_test = percent_corr(nn,x_test,y_test).numpy()
    sys.stdout.write(' time: %s test-acc: %s (error: %s%%)\n' % 
                    (T, round(acc_test,3), round((1-acc_test)*100,3)))