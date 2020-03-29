# import all required libraries

import math
from keras.datasets import mnist
import keras
import random
import tensorflow as tf
from misc import sigmoid, sigmoid_prime
import numpy as np

# class describing the neural network
class Network(object):

  def __init__(self,sizes):

    # number of layers in the neural network
    self.num_layers = len(sizes)

    # size if a list that contains the number of neurons in the respective layers 
    self.sizes = sizes

    self.biases = [tf.random.normal((y,1),mean=0, stddev=1, dtype=tf.dtypes.float32) for y in sizes[1:]]

    self.weights = [tf.random.normal((y,x),mean=0, stddev=1, dtype=tf.dtypes.float32) for x,y in zip(sizes[:],sizes[1:])]

  # this code has problems since we do not have sigmoid for 2-d array
  # search for some way of finding exponent element wise
  # to give the output of the network given some input
  def feedForward(self,a):

    # a is the input
    for b,w in zip(self.biases,self.weights):
      a = sigmoid(tf.tensordot(w,a,1)+b) # activation of next layer from previous layer
    return a

  def evaluate(self, test_data):
    """Return the number of test inputs for which the neural
    network outputs the correct result. The neural
    network's output is assumed to be the index of whichever
    neuron in the final layer has the highest activation."""

    test_results = [(tf.math.argmax(self.feedForward(x)), y) for (x, y) in test_data]
    return sum(int(x == y) for (x, y) in test_results)

  def SGD(self, training_data, epochs, mini_batch_size, eta,test_data=None):

    # eta is the learning rate     
    """Train the neural network using mini-batch stochastic
    gradient descent.  The "training_data" is a list of tuples
    "(x, y)" representing the training inputs and the desired
    outputs.  If "test_data" is provided then the
    network will be evaluated against the test data after each
    epoch, and partial progress printed out.  This is useful for
    tracking progress, but slows things down substantially."""

    if test_data:
      n_test = len(test_data)

    n = len(training_data)
      
    for j in range(epochs):
      random.shuffle(training_data)
      mini_batches = [training_data[k:k+mini_batch_size] for k in range(0, n, mini_batch_size)]

      for mini_batch in mini_batches:
          self.update_mini_batch(mini_batch, eta)

      if test_data:
          print ("Epoch {0}: {1} / {2}".format(j, self.evaluate(test_data), n_test))
      else:
          print ("Epoch {0} complete".format(j))

  def update_mini_batch(self, mini_batch, eta):
    """Update the network's weights and biases by applying
    gradient descent using backpropagation to a single mini batch."""

    # initializing with all zeroes
    nabla_b = [tf.zeros(b.shape, tf.dtypes.float32) for b in self.biases] # nabla means gradient
    nabla_w = [tf.zeros(w.shape, tf.dtypes.float32) for w in self.weights]

    # reflect changes after back propogation
    for x, y in mini_batch:
      delta_nabla_b, delta_nabla_w = self.backprop(x, y)

      nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)] 
      nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]

    # change weights and biases according to gradient descent formula
    self.weights = [w-(eta/len(mini_batch))*nw for w, nw in zip(self.weights, nabla_w)]
    self.biases = [b-(eta/len(mini_batch))*nb for b, nb in zip(self.biases, nabla_b)]


  def backprop(self, x, y):
        """Return a tuple "(nabla_b, nabla_w)" representing the
        gradient for the cost function C_x.  "nabla_b" and
        "nabla_w" are layer-by-layer lists of tensorflow arrays, similar
        to "self.biases" and "self.weights"."""

        nabla_b = [tf.zeros(b.shape, tf.dtypes.float32) for b in self.biases]
        nabla_w = [tf.zeros(w.shape, tf.dtypes.float32) for w in self.weights]

        # feedforward
        activation = x
        activations = [x] # list to store all the activations, layer by layer
        zs = [] # list to store all the z vectors, layer by layer

        for b, w in zip(self.biases, self.weights):
            z = tf.tensordot(w, activation, 1)+b
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)

        # backward pass
        delta = self.cost_derivative(activations[-1], y) * sigmoid_prime(zs[-1])
        nabla_b[-1] = delta
        nabla_w[-1] = tf.tensordot(delta, tf.transpose(activations[-2]), 1)

        #  Here,
        # l = 1 means the last layer of neurons, l = 2 is the
        # second-last layer, and so on. 
        
        for l in range(2, self.num_layers):
            z = zs[-l]
            sp = sigmoid_prime(z)
            delta = tf.tensordot(tf.transpose(self.weights[-l+1]), delta, 1) * sp
            nabla_b[-l] = delta
            nabla_w[-l] = tf.tensordot(delta, tf.transpose(activations[-l-1]), 1)
        return (nabla_b, nabla_w)

  def cost_derivative(self, output_activations, y):
          """Return the vector of partial derivatives \partial C_x /
          \partial a for the output activations."""
          return (output_activations-y) 

  def predict(self, img):
  	test_results = [(self.feedForward(x) for x in img)]
  	return test_results
  	
