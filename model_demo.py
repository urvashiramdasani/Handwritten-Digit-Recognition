# import all required libraries

import math
from keras.datasets import mnist
import random
import tensorflow as tf
from misc import sigmoid, sigmoid_prime
import pickle

# class describing the neural network
class Network(object):

  def __init__(self,sizes):

    # number of layers in the neural network
    self.num_layers = len(sizes)

    # size if a list that contains the number of neurons in the respective layers 
    self.sizes = sizes

    self.biases = [tf.random.normal((y,1),mean=0, stddev=1, dtype=tf.dtypes.float32) for y in sizes[1:]]

    self.weights = [tf.random.normal((y,x),mean=0, stddev=1, dtype=tf.dtypes.float32) for x,y in zip(sizes[:],sizes[1:])]

net = Network([28, 30, 10])
pickle.dump(net, open("demo.sav", "wb"))
# print("hi")
loaded_model = pickle.load(open("demo.sav", "rb"))
print(loaded_model)