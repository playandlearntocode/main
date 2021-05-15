import sys, math
import random
import numpy as np
import autograd.numpy as np
from autograd import grad

# Multilayer Perceptron implementation with autograd
class MLP:
    # NETWORK DIMENSIONS:
    input_layer_size = 3
    hidden_layer_size = 3
    output_layer_size = 2 # output layer / softmax layer number of neurons

    # CONSTRUCTOR:
    def __init__(self, learning_examples_array):
        self.learning_examples_array = learning_examples_array
        self.init_weights()

    def init_weights(self):
        # the list of all weights:
        self.weights = [None, None]

        # first part, between input layer and hidden layer:
        self.weights[0] = np.array([
            # Input layer to hidden layer

            # i1 connections
            [random.random(), random.random(), random.random()],

            # i2 connections
            [random.random(), random.random(), random.random()],

            # i3 connections
            [random.random(), random.random(), random.random()],
        ])

        # second part, between hidden layer and output layer:
        self.weights[1] = np.array([
            # Hidden layer to output layer

            # h1 connections
            [random.random(), random.random()],

            # h2 connections
            [random.random(), random.random()],

            # h3 connections
            [random.random(), random.random()]

        ])

        # FORWARD FEED / PASS - NODE VALUES:
        # important: use 0.0 instead of 0 (otherwise array dtype will be int)

        # weights_gradients structure follows the one used by self.weights.
        self.weights_gradients = [None, None]
        self.weights_gradients[0] = np.array([
            # Input layer to hidden layer

            # i1 connections
            [0.0, 0.0, 0.0],

            # i2 connections
            [0.0, 0.0, 0.0],

            # i3 connections
            [0.0, 0.0, 0.0],
        ])

        self.weights_gradients[1] = np.array([
            # Hidden layer to output layer

            # h1 connections
            [0.0, 0.0],

            # h2 connections
            [0.0, 0.0],

            # h3 connections
            [0.0, 0.0]
        ])

    # logistic function, needed for hidden layer value calculations:
    def activation_function(self,row):
        # return 1.0 / (1.0 + math.exp(-x))
        return 1.0 / (1.0 + np.exp(-row))

    def cross_entropy(self,target_distribution, predicted_distribution):
        return -(np.dot(np.array(target_distribution), np.log(predicted_distribution)))

    # return softmax value for a specified output position:
    def outputs_softmax_loss(self, outputs, correct_output_row):
        total = np.sum(np.exp(outputs))
        singles = np.exp(outputs)
        softmax_row = singles * (1.00 / total)
        # print("SOFTMAX ROW:")
        # print(softmax_row)
        self.prediction_outputs = softmax_row
        # loss_row = (correct_output_row - softmax_row) ** 2
        loss_row = self.cross_entropy(correct_output_row, softmax_row)
        return np.sum(loss_row)

    # PRE: inputs row needs to be filled
    # PRE: correct_outputs row needs to be filled
    def predict(self, weights):
        matmul_hidden = np.matmul(self.inputs, weights[0])
        hidden_layer = self.activation_function(matmul_hidden)
        outputs = np.matmul(hidden_layer, weights[1])
        return self.outputs_softmax_loss(outputs, self.correct_outputs)

    def make_prediction(self, image_object):
        self.inputs = np.array(image_object[1:4], dtype=float)
        self.predict(self.weights)
        return self.prediction_outputs

    def bp_update_weights(self):
        # fixed constant; speed of convergence:
        learning_rate = 0.2

        for layer in range(0, 2):
            for i in range(self.weights[layer].shape[0]):
                for j in range(self.weights[layer].shape[1]):
                    gradient_value = self.weights_gradients[layer][i][j]

                    # move up or down:
                    if (gradient_value > 0):
                        self.weights[layer][i][j] += -learning_rate * abs(gradient_value)
                        # self.weights[layer][i][j] += -learning_rate
                    elif (gradient_value < 0):
                        self.weights[layer][i][j] += learning_rate * abs(gradient_value)
                        # self.weights[layer][i][j] += learning_rate

    # CORE API:
    # take the training input data and update the weights (train the network):
    def train_network(self):
        print('Training network...')

        compute_gradients = grad(self.predict)
        total_loss = 0

        for i in range(0, self.learning_examples_array.shape[0]):
            # real probabilities (target output) for the current training example:
            target_distribution = np.array([self.learning_examples_array[i][4], self.learning_examples_array[i][5]])

            self.inputs = np.array(self.learning_examples_array[i][1:4], dtype=float)
            self.correct_outputs = target_distribution

            current_loss =  self.predict(self.weights)
            # after predict, we also have a new self.prediction_outputs array

            total_loss += current_loss

            self.weights_gradients =  compute_gradients(self.weights)
            self.bp_update_weights()
        return total_loss

    # Calculate the total error on the whole training dataset:
    def calculate_total_error_on_dataset(self, dataset):
        total_delta = 0.0
        total_loss = 0.0

        for i in range(0, self.learning_examples_array.shape[0]):
            #row = self.learning_examples_array[i]
            #target_distribution = np.array([row[4], row[5]])
            current_loss = self.predict(self.weights)
            total_loss += current_loss
        return (-99999, total_loss)

