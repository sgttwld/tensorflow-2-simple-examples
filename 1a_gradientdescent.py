"""
Example of gradient descent to find the minimum of a function using tensorflow 2.1
Author: Sebastian Gottwald
Project: https://github.com/sgttwld/tensorflow-2.0-simple-examples
"""

import tensorflow as tf
import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 

optimizer = tf.keras.optimizers.SGD(learning_rate=.05)

x = tf.Variable(tf.random.normal(shape=(), mean=2.0, stddev=0.5))
f = lambda: (x**2-2)**2

print('init','x =',x.numpy(), 'f(x) =',f().numpy())
for n in range(10):
    optimizer.minimize(f, var_list=[x])
    print('ep',n,'x =',x.numpy(), 'f(x) =',f().numpy())