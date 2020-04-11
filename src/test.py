"""
Main program to load data and save
"""

from keras.datasets import mnist
import tensorflow as tf
from Network import *
import numpy as np

(x_train,y_train), (x_test,y_test) = mnist.load_data()

test_data=[[tf.reshape((tf.convert_to_tensor(x,dtype=tf.float32)),(784,1)),y] for x,y in zip(x_test,y_test)]
train_data=[[tf.reshape((tf.convert_to_tensor(x,dtype=tf.float32)),(784,1)),y] for x,y in zip(x_train,y_train)]

net = Network([784, 30, 10])
net.SGD(train_data, 60, 10, 3.0,test_data)

net.save("model.txt")