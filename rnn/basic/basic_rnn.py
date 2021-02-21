'''
A basic example of a Recurrent Neural Network working with a limited (17 word, includes '.') dictionary in Python from scratch.
Find out more about this code example and many others at https://playandlearntocode.com
Author: Goran Trlin
'''
import re
import time
import numpy as np
import autograd.numpy as np
from autograd import grad
from classes.textinput import TextInput

NUMBER_OF_STEPS = 3  # MEMORY LENGTH / HORIZONTAL LAYERS:

INPUT_LENGTH = 17  # SINGLE WORD IS ENCODED AS 10 BIT ONE-HOT VECTOR
HIDDEN_LENGTH = 17  # NUMBER OF NEURONS IN HIDDEN LAYER PER STEP
OUTPUT_LENGTH = 17  # SAME DIMENSIONS AS INPUT

# FULL VOCABULARY:
VOCABULARY = ['.', 'how', 'are', 'you', 'what', 'is', 'your', 'name', 'where', 'have', 'been', 'go', 'doing',
              'ready', 'now', 'did', 'there']
VOCABULARY_ONE_HOT = [
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
]


def one_hot_to_word(vec1):
    '''
    Converts One-hot vector to word
    :param vec1:
    :return:
    '''
    for i in range(len(VOCABULARY_ONE_HOT)):
        if (np.array_equal(np.array(VOCABULARY_ONE_HOT[i]), np.array(vec1))):
            return VOCABULARY[i]
    return ''


def word_to_one_hot(word):
    '''
    Converts a word to a One-hot vector
    :param word:
    :return:
    '''
    return VOCABULARY_ONE_HOT[VOCABULARY.index(word)]


def parse_input():
    '''
    Reads the input text file containing learning sentences.
    :return:
    '''
    ti = TextInput()
    content = ti.load_text_file('data/learning_examples.txt')
    content_split = re.split(' |\n', content)
    one_hot = []
    strings = []

    for i in range(len(content_split)):
        cur = content_split[i].replace('\n', '')
        # cur = content_split[i].replace('.', '')
        cur = cur.strip()
        cur = cur.lower()

        if (cur == ''):
            continue
        else:
            strings.append(cur)
            one_hot.append(VOCABULARY_ONE_HOT[VOCABULARY.index(cur)])

    return {
        'content': strings,
        'one_hot': one_hot
    }


class BasicRNN:
    # Predicts the next word, so it needs one 17 bit output
    # Reads 3 words and tries to predict the 4th word
    # Type of RNN: hidden units -> hidden units connection
    # 1 word (one-hot vector) as input at every step and 1 word at output (in total)

    # WEIGHTS IN A SINGLE TENSOR:
    weights = [
        np.random.rand(NUMBER_OF_STEPS, INPUT_LENGTH, HIDDEN_LENGTH),
        np.random.rand(NUMBER_OF_STEPS, HIDDEN_LENGTH, OUTPUT_LENGTH)
    ]

    # WEIGHT GRADIENTS (PARAMETERS):
    weights_gradients = [
        np.zeros((NUMBER_OF_STEPS, INPUT_LENGTH, HIDDEN_LENGTH)),
        np.zeros((NUMBER_OF_STEPS, HIDDEN_LENGTH, OUTPUT_LENGTH))
    ]

    # NODE VALUES:
    inputs = np.zeros((NUMBER_OF_STEPS, INPUT_LENGTH))
    hidden_values = np.zeros((NUMBER_OF_STEPS, HIDDEN_LENGTH))

    # 1 NODE RNN OUTPUT:
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

    def make_prediction(self, current_inputs):
        '''
        Callable from main()
        :param current_inputs:
        :return:
        '''
        self.inputs = np.array(current_inputs[:], dtype=float)
        self.predict(self.weights)
        return self.predicted_word

    def predict(self, weights):
        '''
        Core function of this class
        Text is moving to right, while network nodes stays fixed
        :param weights:
        :return:
        '''

        # HIDDEN LAYERS:
        for i in range(NUMBER_OF_STEPS):
            if i == 0:
                # first step:
                current_input_row = self.inputs[i]  # 7x1
                current_weight_matrix = weights[0][i]  # 7x16
                current_row_from_input = np.matmul(np.transpose(current_input_row), current_weight_matrix)
                self.new_hidden_values = np.array([current_row_from_input], dtype=float)
            else:
                # one of the following steps:
                previous_hidden = self
                previous_hidden_row = self.hidden_values[i - 1]  # 16x1
                current_input_row = self.inputs[i]  # 7x1
                current_weight_matrix = weights[0][i]  # 7x16

                current_row_from_input = np.matmul(np.transpose(current_input_row), current_weight_matrix)
                current_row_from_both_sources = current_row_from_input + previous_hidden_row
                self.new_hidden_values = np.concatenate(
                    (self.new_hidden_values, np.array([current_row_from_both_sources], dtype=float)), axis=0)

        # swap/update:
        self.hidden_values = self.new_hidden_values

        # OUTPUT LAYER (SOFTMAX):
        # SINGLE WORD OUTPUT
        current_hidden_row = self.hidden_values[NUMBER_OF_STEPS - 1]  # 16x1
        current_weight_matrix = weights[1][NUMBER_OF_STEPS - 1]

        current_output_row = np.matmul(np.transpose(current_hidden_row), current_weight_matrix)
        current_output_row = np.array(current_output_row, dtype=float)

        current_output_values = self.softmax_function(current_output_row)
        self.output = current_output_values

        friendly_output = np.zeros(OUTPUT_LENGTH)
        friendly_output[np.argmax(self.output)] = 1

        # the next prediction:
        self.predicted_word = friendly_output

        # current loss:
        loss = self.cross_entropy(self.target_distribution, self.output)
        # print('loss=')
        # print(loss)
        return loss

    def bp_update_weights(self):
        '''
        Performs Gradient Descent
        :return:
        '''
        # fixed constant; influces the speed of convergence:
        learning_rate = 0.4

        for layer in range(0, 2):
            for step in range(NUMBER_OF_STEPS):
                for i in range(self.weights[layer][step].shape[0]):
                    for j in range(self.weights[layer][step].shape[1]):
                        gradient_value = self.weights_gradients[layer][step][i][j]

                        # modify the gradient if required:
                        applied_lr = learning_rate
                        applied_gradient = gradient_value

                        # move up or down:
                        if (gradient_value > 0):
                            self.weights[layer][step][i][j] += -applied_lr * abs(applied_gradient)
                            # self.weights[layer][i][j] += -learning_rate
                        elif (gradient_value < 0):
                            self.weights[layer][step][i][j] += applied_lr * abs(applied_gradient)
                            # self.weights[layer][i][j] += learning_rate

    # CORE API:
    # take the training input data and update the weights (train the network):
    def train_network(self, input_string, input_one_hot):
        print('Training network...')
        total_loss = 0

        for i in range(0, len(input_one_hot)):
            if i < NUMBER_OF_STEPS - 1:
                continue

            curIndexStart = i - NUMBER_OF_STEPS
            curIndexEnd = i - 1

            # real probabilities (target output) for the current training example:
            self.target_distribution = np.array(input_one_hot[curIndexEnd + 1], dtype=float)
            label = input_one_hot[i]

            # a bit of regularization (1):
            if (one_hot_to_word(label) == '.'):
                continue

            dot_in_middle = False

            # apply inputs:
            for j in range(NUMBER_OF_STEPS):
                if one_hot_to_word(input_one_hot[curIndexStart + j]) == '.':
                    dot_in_middle = True

                self.inputs[j] = input_one_hot[curIndexStart + j]

            # very basic regularization (2):
            if dot_in_middle == True:
                continue

            loss = self.predict(self.weights)
            # print('loss=')
            # print(loss)

            total_loss += loss
            compute_gradients = grad(self.predict)

            self.weights_gradients = compute_gradients(self.weights)
            self.bp_update_weights()

        return total_loss


# MAIN PROGRAM:
input_object = parse_input()
rnn = BasicRNN()

# TRAINING SETTINGS:
TRAIN_ITERATIONS = 10
TARGET_ACCURACY = 1

total_loss = 99999
total_delta = 9999
train_count = 0

while train_count < TRAIN_ITERATIONS and total_loss > TARGET_ACCURACY:
    total_loss = rnn.train_network(input_object.get('content'), input_object.get('one_hot'))

    print('TOTAL LOSS AT ITERATION (' + str(train_count + 1) + '):')
    print(total_loss)
    train_count += 1
    # time.sleep(5)

print('Training stopped at step #' + str(train_count + 1))

# training completed here
print(input_object.get('content'))

print('*****START OF PREDICTION*****')

current_input_one_hot = [
    word_to_one_hot('what'),
    word_to_one_hot('is'),
    word_to_one_hot('your')

]
current_input_string = [
    'what',
    'is',
    'your'
]

print('INPUT IS:')
print(current_input_string)

print('INPUT ONE-HOT IS:')
print(current_input_one_hot)

current_prediction = rnn.make_prediction(current_input_one_hot)

print('PREDICTED:')
print(current_prediction)

print('WHICH MEANS:')
print(one_hot_to_word(current_prediction))

print('*****END OF PREDICTION*****')

print('*****START OF PREDICTION*****')

current_input_one_hot = [
    word_to_one_hot('are'),
    word_to_one_hot('you'),
    word_to_one_hot('there')

]
current_input_string = [
    'are',
    'you',
    'there'
]

print('INPUT IS:')
print(current_input_string)

print('INPUT ONE-HOT IS:')
print(current_input_one_hot)

current_prediction = rnn.make_prediction(current_input_one_hot)

print('PREDICTED:')
print(current_prediction)

print('WHICH MEANS:')
print(one_hot_to_word(current_prediction))

print('*****END OF PREDICTION*****')

print('*****START OF PREDICTION*****')

current_input_one_hot = [
    word_to_one_hot('did'),
    word_to_one_hot('you'),
    word_to_one_hot('go')

]
current_input_string = [
    'did',
    'you',
    'go'
]

print('INPUT IS:')
print(current_input_string)

print('INPUT ONE-HOT IS:')
print(current_input_one_hot)

current_prediction = rnn.make_prediction(current_input_one_hot)

print('PREDICTED:')
print(current_prediction)

print('WHICH MEANS:')
print(one_hot_to_word(current_prediction))

print('*****END OF PREDICTION*****')
