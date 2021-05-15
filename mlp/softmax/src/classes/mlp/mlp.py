import sys, math
import random
import numpy as np

# Multilayer Perceptron implementation
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
        self.node_values = [None, None, None, None]
        self.node_values[0] = np.array([0.0, 0.0, 0.0])
        self.node_values[1] = np.array([0.0, 0.0, 0.0])
        self.node_values[2] = np.array([0.0, 0.0])
        self.node_values[3] = np.array([0.0, 0.0]) # softmax layer

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

    # FORWARD PASS:
    # set input layer values:
    def ff_apply_inputs(self, image_info_row):
        self.node_values[0][0] = image_info_row[1]
        self.node_values[0][1] = image_info_row[2]
        self.node_values[0][2] = image_info_row[3]
        # add values to input layer:
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

        # now we can calculate softmax layer values:
        for i in range(0, self.output_layer_size):
            self.node_values[3][i] = self.softmax_function(i, self.node_values[2])

    # BACKPROPAGATION:
    # calculate the sensitivity of of weights between the hidden layer and the output layer (1 node in it)_
    def bp_compute_output_layer_gradients(self, target_distribution):
        for k in range(0, self.output_layer_size):
            for i in range(0, self.hidden_layer_size):
                softmax_distribution = self.node_values[3]
                output_values = self.node_values[2]
                sum_exp_softmax = np.sum(np.exp(output_values))
                exp_different_outputs_product = np.prod(np.exp(output_values))
                exp_quotient_value = exp_different_outputs_product / (pow(sum_exp_softmax,2)) * 1.00

                # positive value if the weight is connected to current output, otherwise negative
                if(k == 0):
                    signs = np.array([1,-1])
                elif(k == 1):
                    signs = np.array([-1, 1])

                gradient_value = \
                    - target_distribution[0] * pow(softmax_distribution[0], -1) * \
                    (
                            signs[0] * exp_quotient_value * self.node_values[1][i]
                    ) \
                    - target_distribution[1] * pow(softmax_distribution[1], -1) * \
                    (
                            signs[1] * exp_quotient_value * self.node_values[1][i]
                    )
                self.weights_gradients[1][i][k] = gradient_value


    # gradients between input layer and hidden layer are  smaller than the ones between hidden layer and output layer (problem of vanishing gradient appears):
    def bp_compute_hidden_layer_gradients(self, target_distribution):
        for i in range(0, self.input_layer_size):
            for j in range(0, self.hidden_layer_size):
                # intermediate values:
                column_vector = self.weights[0][:, [j]]
                column_vector = column_vector.transpose()
                input_vector = np.array(self.node_values[0])
                pre_activation_node_value = np.matmul(column_vector, input_vector)[0] # hidden layer node input (before logistic function)

                softmax_distribution = self.node_values[3]
                output_values = self.node_values[2]
                sum_exp_softmax = np.sum(np.exp(output_values))
                exp_different_outputs_product = np.prod(np.exp(output_values))
                exp_quotient_value = exp_different_outputs_product / (pow(sum_exp_softmax,2)) * 1.00

                gradient_value = \
                    - target_distribution[0] * pow(softmax_distribution[0], -1) * \
                    (
                           1 *  exp_quotient_value * self.weights[1][j][0] * self.activation_function_derivative(pre_activation_node_value) * self.node_values[0][i] \
                            - 1 * exp_quotient_value * self.weights[1][j][1] * self.activation_function_derivative(pre_activation_node_value) * self.node_values[0][i]
                    ) \
                    - target_distribution[1] * pow(softmax_distribution[1], -1) * \
                    (
                            -1  * exp_quotient_value * self.weights[1][j][0] * self.activation_function_derivative(pre_activation_node_value) * self.node_values[0][i] \
                                + 1 *  exp_quotient_value  * self.weights[1][j][1] * self.activation_function_derivative(pre_activation_node_value) * self.node_values[0][i]
                    )

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

        for i in range(0, self.learning_examples_array.shape[0]):
            # real probabilities (target output) for the current training example:
            target_distribution = np.array([self.learning_examples_array[i][4], self.learning_examples_array[i][5]])

            # Forward pass:
            self.ff_apply_inputs(self.learning_examples_array[i])
            self.ff_compute_hidden_layer()
            self.ff_compute_output_layer()

            # Backpropagation:
            self.bp_compute_output_layer_gradients(target_distribution)
            self.bp_compute_hidden_layer_gradients(target_distribution)
            self.bp_update_weights()

    # Predict output value for a single input vector (3 features of 1 training or testing example):
    def predict(self, image_object):
        self.ff_apply_inputs(image_object)
        self.ff_compute_hidden_layer()
        self.ff_compute_output_layer()

        softmax_outputs = self.node_values[3]

        return softmax_outputs

    # Calculate the total error on the whole training dataset:
    def calculate_total_error_on_dataset(self, dataset):
        total_delta = 0.0
        total_loss = 0.0

        for i in range(0, self.learning_examples_array.shape[0]):
            row = self.learning_examples_array[i]
            target_distribution = np.array([row[4], row[5]])

            predicted_distribution = self.predict(row)
            current_loss = self.loss_function(target_distribution, predicted_distribution)
            total_loss += current_loss

        return (-99999, total_loss)

    # MATH FUNCTIONS:
    # cross entropy : H(p,q) = -sum[p(x)*log(x)] ; over all ouputs
    def loss_function(self, target_distribution, predicted_distribution):
        cross_entropy_value = - (np.matmul(target_distribution, np.log(predicted_distribution)))
        return cross_entropy_value

    # logistic function, needed for hidden layer value calculations:
    def activation_function(self, x):
        return 1.0 / (1.0 + math.exp(-x))

    # derivative of logistic function g(z)' = g(z) * (1 - g(z)). Needed in backpropagation.
    # to gain a better insight here, write the function compositions on paper, and the find the partial derivatives for all weights.
    def activation_function_derivative(self, x):
        return self.activation_function(x) * (1 - self.activation_function(x))

    # return softmax value for a specified output position:
    def softmax_function(self, output_index, output_values):
        total = np.sum(np.exp(output_values))
        return (math.exp(output_values[output_index])) / total

