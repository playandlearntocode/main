'''
Simple Multilayer Perceptron example in Python
Author: Goran Trlin
Find more tutorials and code samples on:
https://playandlearntocode.com
'''

import numpy
from PIL import Image
from classes.mlp.MLP import MLP
from classes.csv.CsvDataLoader import CsvDataLoader
from classes.extraction.ImageFeatureExtractor import ImageFeatureExtractor

print('MLP program starting...')

#Load training data from CSV files:
csv_data_loader = CsvDataLoader()
data_circle = csv_data_loader.get_training_data('./../csv/correct_outputs_circle.txt')
data_line = csv_data_loader.get_training_data('./../csv/correct_outputs_line.txt')

mlps = [MLP(data_circle), MLP(data_line)]
# mlps[0] - circle detector
# mlps[1] - line detector

# TRAINING PHASE:

# BACKPROPAGATION SETTINGS:
TRAIN_ITERATIONS = 100
TARGET_ACCURACY = 2

for i in range(0, len(mlps)):

    total_loss = 99999
    total_delta = 9999
    train_count = 0

    while train_count < TRAIN_ITERATIONS and total_loss > TARGET_ACCURACY:
        mlps[i].train_network()
        (total_delta, total_loss) = mlps[i].calculate_total_error_on_dataset(mlps[i].learning_examples_array)

        print('TOTAL LOSS AT ITERATION (' + str(i) + '):')
        print(total_loss)
        train_count += 1

    print ('Training stopped at step #' + str(train_count) + ' for i=' + str(i))

# TESTING PHASE:
test_file_name = ''
print('Enter file name to classify:')
test_file_name = input('enter the filename:')

test_file_path = './../test_images/' + test_file_name

image = Image.open(test_file_path)
image.show()

img_extractor = ImageFeatureExtractor()
(im_test_image, test_image_pixels) = img_extractor.load_image(test_file_path)
(test_image_f1, test_image_f2, test_image_f3) = img_extractor.extract_features(im_test_image, test_image_pixels)

test_image_row = [test_file_name, test_image_f1, test_image_f2, test_image_f3, 99999]

output_from_circle = mlps[0].predict(test_image_row)
output_from_line = mlps[1].predict(test_image_row)

print('output from circle detector:')
print(output_from_circle)

print('output from line detector:')
print(output_from_line)

if(output_from_circle>output_from_line):
    print('This image is a CIRCLE')
else:
    print('This image is a LINE')

print('MLP program completed.')