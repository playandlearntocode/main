import sys, math
import random
import numpy as np


class MLP:
    input_layer_size = 3
    hidden_layer_size = 3
    output_layer_size = 1

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
            [random.random()],

            # h2 connections
            [random.random()],

            # h3 connections
            [random.random()]

        ])

        # FORWARD FEED / PASS - NODE VALUES:
        # important: use 0.0 instead of 0 (otherwise array dtype will be int)
        self.node_values = [None, None, None]
        self.node_values[0] = np.array([0.0, 0.0, 0.0])
        self.node_values[1] = np.array([0.0, 0.0, 0.0])
        self.node_values[2] = np.array([0.0])

        # weights_gradients structure follows the one used by self.weights.
        self.weights_gradients = [None, None]
        self.weights_gradients[0] = np.array([
            # Input layer to hidden layer

            # i1 connections
            [0.0, 0.0, 0.0, 0.0],

            # i2 connections
            [0.0, 0.0, 0.0, 0.0],

            # i3 connections
            [0.0, 0.0, 0.0, 0.0],
        ])

        self.weights_gradients[1] = np.array([
            # Hidden layer to output layer

            # h1 connections
            [0.0],

            # h2 connections
            [0.0],

            # h3 connections
            [0.0],

            # h4 connections
            [0.0]

        ])

    # FORWARD FEED:

    # set input layer values:
    def ff_apply_inputs(self, image_info_row):
        self.node_values[0][0] = image_info_row[1]
        self.node_values[0][1] = image_info_row[2]
        self.node_values[0][2] = image_info_row[3]

        self.node_values[0] = np.array([image_info_row[1], image_info_row[2], image_info_row[3]])

    # forward pass - compute hidden layer node values:
    def ff_compute_hidden_layer(self):
        for i in range(0, self.hidden_layer_size):
            column_vector = self.weights[0][:, [i]]
            column_vector = column_vector.transpose()

            new_node_value = np.matmul(column_vector, np.array(self.node_values[0]))[0]
            new_node_value = self.activation_function(new_node_value)
            self.node_values[1][i] = new_node_value

    # forward pass - compute output layer value:
    def ff_compute_output_layer(self):
        for i in range(0, self.output_layer_size):
            column_vector = self.weights[1][:, [i]]
            column_vector = column_vector.transpose()

            # update output node value:
            new_node_value = np.matmul(column_vector, self.node_values[1])[0]
            self.node_values[2][i] = new_node_value

    # basic difference between target ouput value and the obtained (computed) output value:
    def compute_delta(self, target_value, computed_value):
        return (target_value - computed_value)

    # BACKPROPAGATION:
    # calculate the sensitivity of of weights between the hidden layer and the output layer (1 node in it)_
    def bp_compute_output_layer_gradients(self, target_value, output_value_index):
        # if you do the math on paper, you will see that we need the delta value to compute all the gradients:
        delta = self.compute_delta(target_value, self.node_values[2][output_value_index])

        for i in range(0, self.hidden_layer_size):
            gradient_value = - 2 * delta * self.node_values[1][i]
            self.weights_gradients[1][i][output_value_index] = gradient_value


    # gradients between input layer and hidden layer are  smaller than the ones between hidden layer and output layer (problem of vanishing gradient appears):
    def bp_compute_hidden_layer_gradients(self, target_value, output_value_index):
        delta = self.compute_delta(target_value, self.node_values[2][output_value_index])

        for i in range(0, self.input_layer_size):
            for j in range(0, self.hidden_layer_size):
                column_vector = self.weights[0][:, [j]]
                column_vector = column_vector.transpose()
                input_vector = np.array(self.node_values[0])
                pre_activation_node_value = np.matmul(column_vector, input_vector)[0]

                current_input_val = self.node_values[0][i]
                gradient_value = - 2 * delta * self.node_values[1][j]

                gradient_value = gradient_value * self.activation_function_derivative(
                    pre_activation_node_value) * current_input_val

                self.weights_gradients[0][i][j] = gradient_value

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
        print('learning examples array:')
        for i in range(0, self.learning_examples_array.shape[0]):
            target_value = self.learning_examples_array[i][4]
            # Forward pass:

            self.ff_apply_inputs(self.learning_examples_array[i])
            self.ff_compute_hidden_layer()
            self.ff_compute_output_layer()

            for j in range(0, self.output_layer_size):
                self.bp_compute_output_layer_gradients(target_value, j)
                self.bp_compute_hidden_layer_gradients(target_value, j)

            self.bp_update_weights()

    # Predict output value for a single input vector (3 features of 1 training or testing example):
    def predict(self, image_object):
        self.ff_apply_inputs(image_object)
        self.ff_compute_hidden_layer()
        self.ff_compute_output_layer()

        output_value = self.node_values[2][0]
        return output_value

    # Calculate the total error on the whole training dataset:
    def calculate_total_error_on_dataset(self, dataset):
        total_delta = 0.0
        total_loss = 0.0

        for i in range(0, self.learning_examples_array.shape[0]):
            row = self.learning_examples_array[i]
            target_value = row[4]

            predicted_value = self.predict(row)
            total_delta += self.compute_delta(target_value, predicted_value)
            total_loss += self.loss_function(target_value, predicted_value)

        return (total_delta, total_loss)

    # MATH FUNCTIONS:

    # squared(target output - computed output)
    def loss_function(self, target_value, computed_value):
        return math.pow(target_value - computed_value, 2)

    # logistic function, needed for hidden layer value calculations:
    def activation_function(self, x):
        return 1.0 / (1.0 + math.exp(-x))

    # derivative of logistic function g(z)' = g(z) * (1 - g(z)). Needed in backpropagation.
    # to gain a better insight here, write the function compositions on paper, and the find the partial derivatives for all weights.
    def activation_function_derivative(self, x):
        return self.activation_function(x) * (1 - self.activation_function(x))
