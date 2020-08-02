import network
import mnist_loader

training_data, validation_data, test_data = mnist_loader.load_data_wrapper()

activation_equation = network.Relu()

net = network.Network([784, 30, 10], activation_equation)

epochs = 30

mini_batch_size = 10

learning_rate = 0.001

net.stochastic_gradient_descent(training_data, epochs, mini_batch_size, learning_rate, test_data=test_data)