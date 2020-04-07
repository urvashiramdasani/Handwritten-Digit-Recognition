import mnist_loader
from Network import *
from network2 import *

training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
# net = Network([784, 30, 10])
# net.SGD(training_data, 30, 10, 3.0, test_data=test_data)

net = Network([784, 30, 10], cost=CrossEntropyCost)
net.large_weight_initializer()
net.SGD(training_data, 30, 10, 0.5, evaluation_data=test_data, monitor_evaluation_accuracy=True)

net.save("model.txt")