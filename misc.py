# import all required libraries

import math
from keras.datasets import mnist
import random
import tensorflow as tf

# Miscellaneous functions
# to calculate sigmoid function for the next layer
def sigmoid(z):

  return 1.0/(1.0+tf.math.exp(z))

  # if(type(z) == int):
  #   return 1.0/(1.0+math.exp(z))
  # else:
  #   exp_z = []
  #   for i in z:
  #     exp_z.append(1.0/(1.0+math.exp(i)))
  #   return exp_z

def sigmoid_prime(z):
  """Derivative of the sigmoid function."""
  return sigmoid(z)*(1-sigmoid(z))

def feedForward_test(a, weights, biases):

    # print("hi")
    for b,w in zip(biases, weights):
      a = sigmoid(tf.tensordot(w,a,1)+b)
    return a
