'''
SkipGram model
'''

import numpy as np
import autograd.numpy as np
from autograd import grad

from .config import Config
from .text_preprocessor import TextPreprocessor


class SkipGram:
    '''
    Skip-gram algorithm
    '''

    CONFIG = Config()
    text_proc = TextPreprocessor()

    # WEIGHTS IN A SINGLE TENSOR:
    weights = [
        np.random.rand(CONFIG.INPUT_LENGTH, CONFIG.HIDDEN_LENGTH),
        np.random.rand(CONFIG.HIDDEN_LENGTH, CONFIG.OUTPUT_LENGTH)
    ]

    # WEIGHT GRADIENTS (PARAMETERS):
    weights_gradients = [
        np.zeros((CONFIG.INPUT_LENGTH, CONFIG.HIDDEN_LENGTH)),
        np.zeros((CONFIG.HIDDEN_LENGTH, CONFIG.OUTPUT_LENGTH))
    ]

    # NODE VALUES:
    inputs = np.zeros(CONFIG.INPUT_LENGTH)
    hidden_values = np.zeros(CONFIG.HIDDEN_LENGTH)

    output = None

    # NEXT PREDICTION:
    predicted_word = None

    def softmax_function(self, units):
        '''
        Softmax function applied at the output layer
        :param units:
        :return:
        '''
        singles = np.exp(units)
        total = np.sum(np.exp(units))

        softmax_row = singles * (1.00 / total)
        return softmax_row

    def cross_entropy(self, target_distribution, predicted_distribution):
        '''
        Cross-entropy is used for loss calculation
        :param target_distribution:
        :param predicted_distribution:
        :return:
        '''
        sum1 = -(
            np.dot(np.array(target_distribution, dtype=float), np.log(np.array(predicted_distribution, dtype=float))))
        return sum1

    def make_prediction(self, current_input):
        '''
        Callable from main()
        :param current_inputs:
        :return:
        '''
        self.input = np.array(current_input[:], dtype=float)
        self.predict(self.weights)
        return self.predicted_word

    def predict(self, weights):
        '''
        Run through all valid input-output word pairs for 1 input word and calculate the loss
        :param weights:
        :return:
        '''
        loss = 0.00

        for j in range(len(self.correct_output_words)):
            # one comparison
            cur_output_word = self.correct_output_words[j]
            cur_output_one_hot = self.text_proc.word_to_one_hot(cur_output_word, self.CONFIG.VOCABULARY,
                                                                self.CONFIG.VOCABULARY_ONE_HOT)
            self.target_distribution = np.array(cur_output_one_hot, dtype=float)

            # HIDDEN LAYER:
            current_input_row = self.input  # 17x1
            current_weight_matrix = weights[0]  # 17x17
            transposed_weights = np.transpose(current_weight_matrix)
            current_row_from_input = np.matmul( transposed_weights, current_input_row)
            self.hidden_values = np.array(current_row_from_input, dtype=float)

            # OUTPUT LAYER (SOFTMAX):
            # SINGLE WORD OUTPUT
            current_hidden_row = self.hidden_values
            current_weight_matrix = weights[1]

            current_output_row = np.matmul(current_hidden_row, current_weight_matrix)
            current_output_row = np.array(current_output_row, dtype=float)

            self.output = self.softmax_function(current_output_row)

            # current loss:
            loss += self.cross_entropy(self.target_distribution, self.output)
            # print('loss=')
            # print(loss)
        return loss

    def bp_update_weights(self):
        '''
        Performs Gradient Descent
        :return:
        '''
        # hyper parameter of the backpropagation algorithm; this value influences the speed of convergence:
        learning_rate = 0.10

        for layer in range(0, 2):  # 0 -> input to hidden layer; 1 -> hidden layer to output layer
            for i in range(self.weights[layer].shape[0]):
                for j in range(self.weights[layer].shape[1]):
                    gradient_value = self.weights_gradients[layer][i][j]

                    # modify the gradient if required:
                    applied_lr = learning_rate
                    applied_gradient = gradient_value

                    # move up or down:
                    if (gradient_value > 0):
                        self.weights[layer][i][j] += -applied_lr * abs(applied_gradient)
                        # self.weights[layer][i][j] += -learning_rate
                    elif (gradient_value < 0):
                        self.weights[layer][i][j] += applied_lr * abs(applied_gradient)
                        # self.weights[layer][i][j] += learning_rate

    # CORE API:
    # take the training input data and update the weights (train the network):
    def train_network(self, input_strings, input_one_hots):
        print('Training network...')
        total_loss = 0

        for i in range(0, len(input_strings)):

            cur_input_word = input_strings[i]
            cur_input_one_hot = self.text_proc.word_to_one_hot(cur_input_word, self.CONFIG.VOCABULARY,
                                                               self.CONFIG.VOCABULARY_ONE_HOT)
            correct_output_words = self.CONFIG.CORRECT_CONTEXTS.get(cur_input_word)

            if correct_output_words == None or len(correct_output_words) == 0:
                continue

            # apply inputs:
            self.input = cur_input_one_hot
            self.correct_output_words = correct_output_words

            loss = self.predict(self.weights)
            # print('loss=')
            # print(loss)
            total_loss += loss

            compute_gradients = grad(self.predict)
            self.weights_gradients = compute_gradients(self.weights)
            the_weight_gradients = self.weights_gradients[0]
            self.bp_update_weights()



        return total_loss

    def extract_embeddings(self):
        embeddings_dict = {}

        for i in range(len(self.CONFIG.VOCABULARY)):
            cur_word = self.CONFIG.VOCABULARY[i]
            cur_word_one_hot = self.text_proc.word_to_one_hot(cur_word, self.CONFIG.VOCABULARY, self.CONFIG.VOCABULARY_ONE_HOT)
            if cur_word == '.':
                continue

            #cur_pos = len(self.CONFIG.VOCABULARY) -1 -i
            cur_pos = i
            cur_embed =  self.weights[0][cur_pos]
            embeddings_dict[cur_word] = cur_embed

        return embeddings_dict
