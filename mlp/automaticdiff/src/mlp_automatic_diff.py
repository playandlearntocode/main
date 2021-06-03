'''
Simple Multilayer Perceptron with Softmax and Cross-Entropy. Backpropagation handled using autograd package (automatic differentiation).
Author: Goran Trlin
Find more tutorials and code samples on:
https://playandlearntocode.com
'''
import numpy
from PIL import Image
from classes.mlp.mlp import MLP
from classes.csv.csv_data_loader import CsvDataLoader
from classes.extraction.image_feature_extractor import ImageFeatureExtractor

print('MLP program starting...')

#Load training data from CSV files:
csv_data_loader = CsvDataLoader()
data_training =  csv_data_loader.get_training_data('./../csv/correct_distributions_for_training.txt')

mlps = [MLP(data_training)]

# TRAINING PHASE:

# BACKPROPAGATION SETTINGS:
TRAIN_ITERATIONS = 100
TARGET_ACCURACY = 2

for i in range(0, len(mlps)):
    total_loss = 99999
    total_delta = 9999
    train_count = 0

    while train_count < TRAIN_ITERATIONS and total_loss > TARGET_ACCURACY:
        total_loss = mlps[i].train_network()

        print('TOTAL LOSS AT ITERATION (' + str(train_count) + '):')
        print(total_loss)
        train_count += 1

    print ('Training stopped at step #' + str(train_count) + ' for i=' + str(i))

# TESTING PHASE:
test_file_name = input('enter the filename:')
test_file_path = './../test_images/' + test_file_name

image = Image.open(test_file_path)
image.show()

img_extractor = ImageFeatureExtractor()
(im_test_image, test_image_pixels) = img_extractor.load_image(test_file_path)
(test_image_f1, test_image_f2, test_image_f3) = img_extractor.extract_features(im_test_image, test_image_pixels)

test_image_row = [test_file_name, test_image_f1, test_image_f2, test_image_f3, 0,0]

print(test_image_row)
output_from_mlp = mlps[0].make_prediction(test_image_row)

print('Output:')
print(output_from_mlp)

labels = ['CIRCLE', 'LINE']

index =  numpy.argmax(output_from_mlp)
label = labels[index]

print ('This image is a ' + label)
print('MLP program completed.')