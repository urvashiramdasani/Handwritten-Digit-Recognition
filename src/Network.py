"""
A simple code for recognizing handwritten digits. We have used stochastic gradient descent algorithm to train 
neural network. This is the one of the algorithm to get the set of minimizing weights and biases for neural network.
The main perpose of this algorithm is to find the lowest point of function. It starts with random point on a function
and travels down to its slope step by step such that it reaches to the minimum point of function. In our program it 
starts with random weights and biases and then it upadates them such that the cost of our network be minimal.
"""

import random
import json
import tensorflow as tf
import numpy as np

class Network(object):

    def __init__(self, sizes):
        """
        Here `sizes` is list of number of neurons in each layer.The lenght of sizes is the number of layers in network.
        If for example sizes=[1,2,3] , that means there are total three layers of neurons in network and first layer
        contains one, second contains 2 and third layer of network contains three neurons
        """
        data=load("../model/model.txt")
        
        self.sizes = sizes
        self.num_layers = len(sizes)
        self.biases = [tf.convert_to_tensor(b) for b in data["biases"]]
        self.weights = [tf.convert_to_tensor(w) for w in data["weights"]]


    def feedforward(self, a):
        """Returns the activation of next layer as an input of a"""
        for b, w in zip(self.biases, self.weights):
            a = sigmoid(tf.tensordot(w,a,axes=1)+b)
        return a

    def SGD(self, training_data, epochs, mini_batch_size, eta,test_data=None):
        """
        SGD is stochastic gradient descent, used to train network to recognize image.`epochs` is the number of epochs
        which is a hyperparameter that defines the number of times this network works through the entire training dataset.
        `mini_batch_size` is size of a single batch. A batch is the sample work of trainig data set to work through befor
        upadating the parameter of network.`eta` is learning rate of network.`test_data` is data to test the network
        """

        if test_data: 
            n_test = len(test_data)

        n = len(training_data)

        for j in range(epochs):
            random.shuffle(training_data)
            mini_batches = [training_data[k:k+mini_batch_size] for k in range(0, n, mini_batch_size)]
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, eta)
            if test_data:
                print ("Epoch {0}: {1} / {2}".format(j+1, n_test-self.evaluate(test_data), n_test))
            else:
                print ("Epoch {0} complete".format(j+1))

    def update_mini_batch(self, mini_batch, eta):
        """
        Upadates weights and biases of network by applying stochastic gradient descent after every single mini batch
        using backpropagation algorithm.
        """
        nabla_b = [tf.zeros((b.shape),tf.float32) for b in self.biases]
        nabla_w = [tf.zeros((w.shape),tf.float32) for w in self.weights]
        for x, y in mini_batch:
            delta_nabla_b, delta_nabla_w = self.backprop(x, y)#backpropagation algorithm to update weights ans biases
            nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
        self.weights = [w-(eta/len(mini_batch))*nw for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b-(eta/len(mini_batch))*nb for b, nb in zip(self.biases, nabla_b)]

    def backprop(self, x, y):
        """
        Return a tuple of nabla_b and nabla_w which is the biases and weights of perticular image `x`. First we give x as 
        input and find the output. Then we compare it with our expected output and find cost. Using this cost we return
        biases and weights for this input as nabla_w and nabla_b respectively.
        """
        nabla_b = [tf.zeros((b.shape),tf.float32) for b in self.biases]
        nabla_w = [tf.zeros((w.shape),tf.float32) for w in self.weights]
        
        #forward pass
        activation = x #activation of input layer i.e. first layer of nerwork
        activations = [x] # list to store all the activations, layer by layer
        zs = [] # list to store all the z vectors, layer by layer

        for b, w in zip(self.biases, self.weights):
            z = tf.tensordot(w, activation,axes=1)+b
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)

        # backward pass
        delta = self.cost_derivative(activations[-1], y) * sigmoid_prime(zs[-1])
        nabla_b[-1] = delta
        nabla_w[-1] = tf.tensordot(delta, tf.transpose(activations[-2]),axes=1)
        """
        Here l=1 means last layer, l=2 means second last layer and so on.
        """
        for l in range(2, self.num_layers):
            z = zs[-l]
            sp = sigmoid_prime(z)
            delta = tf.tensordot(tf.transpose(self.weights[-l+1]), delta,axes=1) * sp
            nabla_b[-l] = delta
            nabla_w[-l] = tf.tensordot(delta, tf.transpose(activations[-l-1]),axes=1)
        return (nabla_b, nabla_w)

    def evaluate(self, test_data):
        """
        Return the number of test inputs for which the neural network outputs the correct result i.e the index
        of last layer of neuron which has higher activation value.
        """
        test_results = [(tf.math.argmax(self.feedforward(x)), y) for (x, y) in test_data]
        return sum(int(x == y) for (x, y) in test_results)

    def cost_derivative(self, output_activations, y):
        """Return the cost which is the difference between output activations value and expacted value"""
        return (output_activations-y)

    def save(self, filename):
        """Save the neural network to the file ``filename``."""
        data = {"sizes": self.sizes,
                "weights": [np.array(w).tolist() for w in self.weights],
                "biases": [np.array(b).tolist() for b in self.biases]}
        f = open(filename, "w")
        json.dump(data, f)
        f.close()

def load(filename):
    """Load a neural network from the file ``filename``.  Returns an
    instance of Network."""
    f = open(filename, "r")
    data = json.load(f)
    f.close()
    return data

#### Miscellaneous functions
def sigmoid(z):
    """The sigmoid function to set activations fo layers"""
    return 1.0/(1.0+np.exp(-z))

def sigmoid_prime(z):
    """Derivative of the sigmoid function."""
    return sigmoid(z)*(1-sigmoid(z))
