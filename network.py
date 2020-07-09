"""
network.py
~~~~~~~~~~

A module to implement the stochastic gradient descent learning
algorithm for a feedforward neural network.  Gradients are calculated
using backpropagation.  Note that I have focused on making the code
simple, easily readable, and easily modifiable.  It is not optimized,
and omits many desirable features.
"""

#### Libraries
# Standard library
import random

# Third-party libraries
import numpy as np

class Network(object):

    def __init__(self, sizes: list):
        """The list ``sizes`` contains the number of neurons in the
        respective layers of the network.  For example, if the list
        was [2, 3, 1] then it would be a three-layer network, with the
        first layer containing 2 neurons, the second layer 3 neurons,
        and the third layer 1 neuron.  The biases and weights for the
        network are initialized randomly, using a Gaussian
        distribution with mean 0, and variance 1.  Note that the first
        layer is assumed to be an input layer, and by convention we
        won't set any biases for those neurons, since biases are only
        ever used in computing the outputs from later layers."""

        self.num_layers = len(sizes)
        self.sizes = sizes
        self.biases = self.initialize_biases(sizes)
        self.weights = self.initialize_weights(sizes)

    def initialize_biases(self, layer_sizes: list) -> list:
        """
        Returns a list of lists of biases for each neuron in each layer
        after the input layer.
        """

        biases = []

        # Loop through the every layer after the first one
        for size in layer_sizes[1:]:

            # Generate a random bias for each neuron in the layer
            bias = np.random.randn(size, 1)

            biases.append(bias)

        return biases

    def initialize_weights(self, layer_sizes: list) -> list:
        """
        Returns a list of lists of weights for each neuron in each layer.
        """

        weights = []

        # Loop through a pair of layer n and layer n+1 tuples
        # This represents the weights between the layers
        for x, y in zip(layer_sizes[:-1], layer_sizes[1:]):

            # Generate a set of weights that is n+1 layer x n layer
            weight = np.random.randn(y, x)

            weights.append(weight)

        return weights

    def feedforward(self, activation):
        """Return the output of the network if ``a`` is input."""

        for bias, weight in zip(self.biases, self.weights):

            activation = sigmoid(np.dot(weight, activation) + bias)

        return activation

    def SGD(self, training_data, epochs: int, mini_batch_size: int, learning_rate: float, test_data=None) -> None:
        """Train the neural network using mini-batch stochastic
        gradient descent.  The ``training_data`` is a list of tuples
        ``(features, labels)`` representing the training inputs and the desired
        outputs.  The other non-optional parameters are
        self-explanatory.  If ``test_data`` is provided then the
        network will be evaluated against the test data after each
        epoch, and partial progress printed out.  This is useful for
        tracking progress, but slows things down substantially."""

        if test_data: 
            num_test = len(test_data)

        for epoch in range(epochs):

            random.shuffle(training_data)

            mini_batches = self.create_mini_batches(mini_batch_size, training_data)

            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, learning_rate)

            if test_data:
                print("Epoch {0}: {1} / {2}".format(epoch, self.evaluate(test_data), num_test))
            else:
                print("Epoch {0} complete".format(epoch))


    def create_mini_batches(self, mini_batch_size: int, training_data: list) -> list:
        """
        Divides the training data into mini batches of size mini_batch_train 

        Args:
            mini_batch_size: Size of each mini batch
            training_data: The data the model is being trained on

        Returns:
            mini_batches: All of the mini batches created from the training data
        """

        num_train = len(training_data)

        mini_batches = []

        for batch_start in range(0, num_train, mini_batch_size):
            
            batch_end = batch_start + mini_batch_size

            batch = training_data[batch_start:batch_end]

            mini_batches.append(batch)

        return mini_batches



    def update_mini_batch(self, mini_batch: list, learning_rate: float) -> None:
        """
        Update the network's weights and biases by applying gradient descent using backpropagation to a single mini batch.
        The ``mini_batch`` is a list of tuples ``(x, y)``, and ``learning_rate`` is the learning rate.

        Args:
            mini_batch: One of the mini batches from the training data, a list of features and labels
            learning_rate: The learning rate to be used for training
        """

        nabla_b = [np.zeros(b.shape) for b in self.biases]

        nabla_w = [np.zeros(w.shape) for w in self.weights]

        for x, y in mini_batch:

            delta_nabla_b, delta_nabla_w = self.backprop(x, y)

            nabla_b = [nb + dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]

            nabla_w = [nw + dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]

        self.weights = [w - (learning_rate / len(mini_batch)) * nw for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b - (learning_rate / len(mini_batch)) * nb for b, nb in zip(self.biases, nabla_b)]

    def backprop(self, features: list, labels: list) -> tuple:
        """
        Return a tuple ``(nabla_b, nabla_w)`` representing the gradient for the cost function C_x.  
        ``nabla_b`` and ``nabla_w`` are layer-by-layer lists of numpy arrays, similar to ``self.biases`` 
        and ``self.weights``.

        Args:
            features: The input to the network
            labels: The expected output

        Returns:
            gradient: The gradient for the cost function C_x
        """

        nabla_b = [np.zeros(b.shape) for b in self.biases]

        nabla_w = [np.zeros(w.shape) for w in self.weights]

        # feedforward
        activation = features
        activations = [features] # list to store all the activations, layer by layer

        zs = [] # list to store all the z vectors, layer by layer

        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation)+b
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)

        # backward pass
        delta = self.cost_derivative(activations[-1], labels) * sigmoid_prime(zs[-1])

        nabla_b[-1] = delta

        nabla_w[-1] = np.dot(delta, activations[-2].transpose())

        # Note that the variable l in the loop below is used a little
        # differently to the notation in Chapter 2 of the book.  Here,
        # l = 1 means the last layer of neurons, l = 2 is the
        # second-last layer, and so on.  It's a renumbering of the
        # scheme in the book, used here to take advantage of the fact
        # that Python can use negative indices in lists.

        for l in range(2, self.num_layers):
            z = zs[-l]
            sp = sigmoid_prime(z)
            delta = np.dot(self.weights[-l + 1].transpose(), delta) * sp
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l - 1].transpose())

        gradient = (nabla_b, nabla_w)

        return gradient

    def evaluate(self, test_data: list):
        """
        Return the number of test inputs for which the neural network outputs the correct result. Note that the neural
        network's output is assumed to be the index of whichever neuron in the final layer has the highest activation.

        Args:
            test_data: The data to test the network's accuracy on

        Returns:
            num_correct: The number of test data that the network predicted correctly on
        """

        test_results = []

        for (x, y) in test_data:

            result = (np.argmax(self.feedforward(x)), y)

            test_results.append(result)

        num_correct = sum(int(x == y) for (x, y) in test_results)

        return num_correct

    def cost_derivative(self, output_activations, y):
        """Return the vector of partial derivatives partial C_x 
        partial a for the output activations."""

        return (output_activations-y)

#### Miscellaneous functions
def sigmoid(z):
    """The sigmoid function."""
    
    return 1.0 / (1.0 + np.exp(-z))

def sigmoid_prime(z):
    """Derivative of the sigmoid function."""

    return sigmoid(z) * (1 - sigmoid(z))