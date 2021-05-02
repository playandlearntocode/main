import csv
import numpy as np
import pandas
from PIL import Image
from os import walk

class FolderHelper:

    def get_files_list(self, folder_path):
        files_list = []
        for(dir_path, dir_names, file_names) in walk(folder_path):
            files_list.extend(file_names)

        return files_list

    def process_single_image(self, file_path):
        im = Image.open(file_path)
        arr = np.array(im)  # arr.shape is h x w x channels
        return arr

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