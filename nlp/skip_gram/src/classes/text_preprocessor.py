import re
import autograd.numpy as np

class TextPreprocessor:
    '''
    Reads the input file, converts the strings to one hot vectors and vice versa, builds a dictionary of correct contexts for all input words
    '''

    def load_text_file(self, filepath):
        f = open(filepath, "r")
        return f.read()

    def one_hot_to_word(self, one_hot_vector, vocabulary, vocabulary_one_hot):
        '''
        Converts One-hot vector to word
        :param one_hot_vector:
        :return:
        '''
        for i in range(len(vocabulary_one_hot)):
            if (np.array_equal(np.array(vocabulary_one_hot[i]), np.array(one_hot_vector))):
                return vocabulary[i]
        return ''

    def word_to_one_hot(self, word, vocabulary, vocabulary_one_hot):
        '''
        Converts a word to a One-hot vector
        :param word:
        :return:
        '''
        return vocabulary_one_hot[vocabulary.index(word)]

    def add_to_probabilities(self, input_word, output_word, correct_contexts):
        '''
        Works with strings
        :param input_word:
        :param output_word:
        :param correct_contexts:
        :return:
        '''
        if correct_contexts == None:
            correct_contexts = {}

        if correct_contexts.get(input_word) == None:
            correct_contexts[input_word] = []

        if output_word not in correct_contexts.get(input_word):
            correct_contexts[input_word].append(output_word)

    def parse_input(self, file_path, vocabulary, vocabulary_one_hot):
        '''
        Reads the input text file containing learning sentences.
        :return:
        '''
        content = self.load_text_file(file_path)
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
                one_hot.append(vocabulary_one_hot[vocabulary.index(cur)])

        return {
            'content': strings,
            'one_hot': one_hot
        }

    def process_window(self, window_size, words, correct_contexts):
        '''
        Create valid pairs input word - output word. Returns a dictionary - [input word] - [context words]
        :param window_size:
        :param words:
        :param correct_contexts:
        :return:
        '''
        for i in range(len(words)):
            for j in range(i-window_size, i+window_size+1):
                cur_index = j

                if(cur_index < 0 or (cur_index > len(words) -1)):
                    # invalid index
                    continue

                if i == j:
                    # same word
                    continue

                input_word = words[i]
                output_word = words[j]

                if input_word.strip() == '.':
                    continue

                if output_word.strip() == '.':
                    continue

                self.add_to_probabilities(input_word, output_word, correct_contexts)

        return correct_contexts