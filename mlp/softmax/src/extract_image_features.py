# Extract Image Features and save them to a CSV file
import numpy
from classes.extraction.ProcessImageFolder import ProcessImageFolder
from classes.csv.CsvDataLoader import CsvDataLoader

print('Image feature extraction is starting.')

process_handler = ProcessImageFolder()

#process_handler.process_single_image('./../images/1.png')

process_handler.process_folder('./../images')


print('Image feature extraction completed.')
