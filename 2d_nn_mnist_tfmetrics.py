"""
Example of a simple neural network for MNIST classification 
using the layers API, custom training, and tf.keras.metrics in tensorflow 2.1 
Author: Sebastian Gottwald
Project: https://github.com/sgttwld/tensorflow-2.0-simple-examples
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras import Model, Input
import os, time, math, sys
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 

def print_progress(i, tot, lss, acc, acc_str, bar_length=30, wait=False):
    filled_length = int(round(bar_length * i / tot))
    # bar = '█' * filled_length + '-' * (bar_length - filled_length)
    bar = '|' * filled_length + '-' * (bar_length - filled_length)
    sys.stdout.write('\r%s/%s |%s| %s %s %s %s' % (i, tot, bar, 'loss:', lss, acc_str+':', acc)),
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

# metrics
train_loss = tf.keras.metrics.Mean(name='train_loss')
train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')
test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')

# train function
@tf.function
def train(model, x, y):
    with tf.GradientTape() as g:
        predictions = model(x, training=True)
        loss = loss_fn(y, predictions)
    gradients = g.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    train_loss(loss)
    train_accuracy(y, predictions)

# test function
@tf.function
def test(model, x, y):
  # training=False is only needed if there are layers with different
  # behavior during training versus inference (e.g. Dropout).
  test_accuracy(y, model(x, training=False))

# use tf.data to batch and shuffle the dataset
bs = 32
train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train)).shuffle(10000).batch(bs)
test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(bs)
numBatches = math.floor(len(x_train)/bs)

for epoch in range(5):
    t0 = time.time()
    train_loss.reset_states()
    train_accuracy.reset_states()
    test_accuracy.reset_states()
    
    b = 0
    for batch_X, batch_Y in train_ds:
        train(nn, batch_X, batch_Y)
        acc = train_accuracy.result().numpy()
        lss = train_loss.result().numpy()
        print_progress(b, numBatches, round(lss,3), round(acc,3), acc_str='acc', wait=True)
        b += 1

    for test_X, test_Y in test_ds:
        test(nn, test_X, test_Y)

    # evaluation over episodes
    T = round(time.time()-t0,2)
    acc_test = test_accuracy.result().numpy()
    sys.stdout.write(' time: %s test-acc: %s (error: %s%%)\n' % 
                    (T, round(acc_test,3), round((1-acc_test)*100,3)))