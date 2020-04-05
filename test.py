import mnist_loader
from Network import *

training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
net = Network([784, 30, 10])