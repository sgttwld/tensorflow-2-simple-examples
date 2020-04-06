"""
Example of gradient descent to find the minimum of a function using tensorflow 2.1 
with explicit gradient calculation that allows further processing
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

def opt(f,x):
    with tf.GradientTape() as g:
      y = f()
    grad = g.gradient(y, x)
    ## here we could do some gradient processing
    ## grad = ... 
    optimizer.apply_gradients(zip([grad], [x]))

print('init','x =',x.numpy(), 'f(x) =',f().numpy())

for n in range(10):
    opt(f,x)
    print('ep',n,'x =',x.numpy(), 'f(x) =',f().numpy())