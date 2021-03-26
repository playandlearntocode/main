from scipy import spatial
import autograd.numpy as np
from classes.text_preprocessor import TextPreprocessor
from classes.skip_gram import SkipGram

def cosine_distance(embbedding1, embedding2):
    '''
    Small values indicate similar embeddings ( appearing in many similar contexts )
    High values indicate non-related embeddings (words)
    :param embbedding1:
    :param embedding2:
    :return:
    '''
    return spatial.distance.cosine(embbedding1, embedding2)


# MAIN PROGRAM:
print('***STARTING MAIN PROGRAM***')

nn = SkipGram()
preprocessor = TextPreprocessor()
input_file_path = '../data/text_to_learn_from.txt'
preprocessor.load_text_file(input_file_path)

file_processed =  preprocessor.parse_input(input_file_path, nn.CONFIG.VOCABULARY,nn.CONFIG.VOCABULARY_ONE_HOT)
preprocessor.process_window(nn.CONFIG.WINDOW_LENGTH, file_processed.get('content'), nn.CONFIG.CORRECT_CONTEXTS)

print('Correct contexts w.r.t. target word:')
print(nn.CONFIG.CORRECT_CONTEXTS) # valid pairs of input and output words.

# NEURAL NETWORK TRAINING SETTINGS:
TRAIN_ITERATIONS = 50
TARGET_ACCURACY = 1

total_loss = 99999
total_delta = 9999
train_count = 0

while train_count < TRAIN_ITERATIONS and total_loss > TARGET_ACCURACY:
    total_loss = nn.train_network(file_processed.get('content'), file_processed.get('one_hot'))

    print('TOTAL LOSS AT ITERATION (' + str(train_count + 1) + '):')
    print(total_loss)
    train_count += 1
    # time.sleep(5)

print('Training stopped at step #' + str(train_count + 1))

# extract word embeddings:
embeddings_dict = nn.extract_embeddings()

print('***GENERATED EMBEDDINGS:***')
print(embeddings_dict)

'''
Note: words that end up together in differebt correct_output_words lists ( output lists for different input words ) should have small cosine distance. Example: "go" and "are"
Words that are not related at all (not appearing in the same contexts) should have high cosine distance. Example: "go" and "is"
'''

# COMPARE (SHOULD BE SIMILAR):
w1 = 'go'
w2 = 'now'

w1_embedding = embeddings_dict.get(w1)
w2_embedding = embeddings_dict.get(w2)

print('Cosine distance between ' + w1 + ' and ' + w2 + ':')
print(cosine_distance(w1_embedding, w2_embedding))


# COMPARE (SHOULD BE DISSIMILAR):
w1 = 'go'
w2 = 'is'

w1_embedding = embeddings_dict.get(w1)
w2_embedding = embeddings_dict.get(w2)

print('Cosine distance between ' + w1 + ' and ' + w2 + ':')
print(cosine_distance(w1_embedding, w2_embedding))

print('Probability dictionary:')
print(nn.CONFIG.CORRECT_CONTEXTS)


# COMPARE (SHOULD BE SIMILAR):
w1 = 'what'
w2 = 'your'

w1_embedding = embeddings_dict.get(w1)
w2_embedding = embeddings_dict.get(w2)

print('Cosine distance between ' + w1 + ' and ' + w2 + ':')
print(cosine_distance(w1_embedding, w2_embedding))

print('Probability:')
print(nn.CONFIG.CORRECT_CONTEXTS)


# COMPARE (SHOULD BE DISSIMILAR):
w1 = 'go'
w2 = 'name'

w1_embedding = embeddings_dict.get(w1)
w2_embedding = embeddings_dict.get(w2)

print('Cosine distance between ' + w1 + ' and ' + w2 + ':')
print(cosine_distance(w1_embedding, w2_embedding))

print('Correct contexts again:')
print(nn.CONFIG.CORRECT_CONTEXTS)

print('***MAIN PROGRAM COMPLETED***')