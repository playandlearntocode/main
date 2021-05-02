# This class takes a folder containing images and extracts features of each image. At the end of the process, the features  can be saved to a .csv file
import csv
import numpy as np
import pandas
from os import walk
from .image_helper import ImageHelper

class FolderHelper:

    def get_files_list(self, folder_path):
        files_list = []

        for(dir_path, dir_names, file_names) in walk(folder_path):
            files_list.extend(file_names)

        return files_list

    def process_single_image(self, file_path):
        image_feature_extractor = ImageHelper()
        return image_feature_extractor.load_image(file_path)


    def process_images_folder(self, folder_path,width,height):
        files_list = self.get_files_list(folder_path)
        data = np.zeros((len(files_list),height,width,3))
        counter = 0

        for file_name in files_list:
            file_path = folder_path + '/' + file_name

            table = self.process_single_image(file_path)
            data[counter] = table
            counter+=1

        return data

    def process_labels_file(self, file_path):
        csv_content = pandas.read_csv(file_path)
        return csv_content