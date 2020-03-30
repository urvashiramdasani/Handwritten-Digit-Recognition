# main program

# import all required libraries

import math
from keras.datasets import mnist
import random
import tensorflow as tf
from misc import sigmoid, sigmoid_prime
from Network import *
import pickle

(x_train,y_train), (x_test,y_test) = mnist.load_data()

x_validation = x_train[50000:]
y_validation = y_train[50000:]

x_train = x_train[:50000]
y_train = y_train[:50000]

test_data = list()
training_data = list()
validation_data = list()

for x,y in zip(x_test,y_test):
  test_data.append((tf.convert_to_tensor(x,tf.dtypes.float32),y))

for x,y in zip(x_train,y_train):
  training_data.append((tf.convert_to_tensor(x,tf.dtypes.float32),y))
   
for x,y in zip(x_validation,y_validation):
  validation_data.append((tf.convert_to_tensor(x,tf.dtypes.float32),y))

net = Network([28, 30, 10])
net.SGD(training_data, 30, 10, 3.0)

print("The model has been successfully trained.")

print("Saving model into mnist.txt file...")


weights = np.array(net.weights)
biases = np.array(net.biases)

pickle.dump(net.weights, open("weights.txt", "wb"))
pickle.dump(net.biases, open("biases.txt", "wb"))

print("Model saved successfully!")
