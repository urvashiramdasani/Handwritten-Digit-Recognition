"""
Main program to load data and save
"""

from keras.datasets import mnist
import tensorflow as tf
from Network import *
import numpy as np

#Getting the training and testing datasets from mnist dataset

(x_train,y_train), (x_test,y_test) = mnist.load_data()

test_data=[[tf.reshape((tf.convert_to_tensor(x,dtype=tf.float32)),(784,1)),y] for x,y in zip(x_test,y_test)]
train_data=[[tf.reshape((tf.convert_to_tensor(x,dtype=tf.float32)),(784,1)),y] for x,y in zip(x_train,y_train)]

"""
Our network has 784 neurons in first, 30 neurons in second and 10 neurons in last layer.
"""

net = Network([784, 30, 10])

"""
Traing the network whith 30 number of epochs, size of minibatch is 10 and with learning rate of 3.0
"""

net.SGD(train_data, 60, 10, 3.0,test_data)

print("Saving the neural network to model.txt file..... ")
net.save("model.txt")
