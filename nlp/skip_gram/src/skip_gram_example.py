import autograd.numpy as np
from classes.text_preprocessor import TextPreprocessor
from classes.skip_gram import SkipGram

def cosine_similarity(embbedding1, embedding2):
    '''
    High values indicate similar embeddings ( appearing in many similar contexts )
    Small values indicate non-related embeddings (words)
    :param embbedding1:
    :param embedding2:
    :return:
    '''

    cos_sim = np.dot(embbedding1, embedding2) / (np.linalg.norm(embbedding1) * np.linalg.norm(embedding2))
    return cos_sim

# MAIN PROGRAM:Î
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
Note: words that end up together in differebt correct_output_words lists ( output lists for different input words ) should have small cosine distance (high cosine similarity). Example: "go" and "are"
Words that are not related at all (not appearing in the same contexts) should have high cosine distance (small cosine similarity). Example: "go" and "is"
'''

# COMPARE (SHOULD BE SIMILAR):
w1 = 'go'
w2 = 'now'

w1_embedding = embeddings_dict.get(w1)
w2_embedding = embeddings_dict.get(w2)

print('Cosine similarity between ' + w1 + ' and ' + w2 + ' (should be high):')
print(cosine_similarity(w1_embedding, w2_embedding))


# COMPARE (SHOULD BE DISSIMILAR):
w1 = 'go'
w2 = 'is'

w1_embedding = embeddings_dict.get(w1)
w2_embedding = embeddings_dict.get(w2)

print('Cosine similarity between ' + w1 + ' and ' + w2 + ' (should be low):')
print(cosine_similarity(w1_embedding, w2_embedding))

# COMPARE (SHOULD BE SIMILAR):
w1 = 'what'
w2 = 'your'

w1_embedding = embeddings_dict.get(w1)
w2_embedding = embeddings_dict.get(w2)

print('Cosine similarity between ' + w1 + ' and ' + w2 + ' (should be high):')
print(cosine_similarity(w1_embedding, w2_embedding))

# COMPARE (SHOULD BE DISSIMILAR):
w1 = 'go'
w2 = 'name'

w1_embedding = embeddings_dict.get(w1)
w2_embedding = embeddings_dict.get(w2)

print('Cosine similarity between ' + w1 + ' and ' + w2 + ' (should be low):')
print(cosine_similarity(w1_embedding, w2_embedding))

print('Correct contexts again:')
print(nn.CONFIG.CORRECT_CONTEXTS)

print('***MAIN PROGRAM COMPLETED***')