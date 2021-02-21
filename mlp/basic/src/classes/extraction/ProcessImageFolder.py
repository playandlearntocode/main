# This class takes a folder containing images and extracts features of each image. At the end of the process, the features  can be saved to a .csv file
import csv
from os import walk
from classes.extraction.ImageFeatureExtractor import ImageFeatureExtractor


class ProcessImageFolder:

    def get_files_list(self, folder_path):
        files_list = []

        for(dir_path, dir_names, file_names) in walk(folder_path):
            files_list.extend(file_names)

        return files_list

    def process_single_image(self, file_path):
        image_feature_extractor = ImageFeatureExtractor()
        (image, pixels) = image_feature_extractor.load_image(file_path)
        return image_feature_extractor.extract_features(image,pixels)

    def process_folder(self, folder_path):
        files_list = self.get_files_list(folder_path)


        with open ('./../csv/input-file.txt',mode='w',newline='') as csv_file:
            file_writer = csv.writer(csv_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            file_writer.writerow(['file_name', 'feature1', 'feature2', 'feature3'])
            for file_name in files_list:
                file_path = folder_path + '/' + file_name

                (feature1, feature2, feature3) = self.process_single_image(file_path)
                file_writer.writerow([file_name, feature1, feature2, feature3])
