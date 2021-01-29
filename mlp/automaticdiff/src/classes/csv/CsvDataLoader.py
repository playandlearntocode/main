import csv
import numpy
import pandas

class CsvDataLoader:

    def load_image_features(self, file_path):
        rows = []
        count = 0
        return pandas.read_csv('./../csv/input-file.txt')



    def load_correct_outputs(self, file_path):
        rows = []
        count = 0
        return pandas.read_csv(file_path)



    def get_training_data(self, file_path):
        image_features_df = self.load_image_features('./../csv/input-file.txt')
        correct_outputs_df =  self.load_correct_outputs(file_path)

        training_df = pandas.merge(image_features_df, correct_outputs_df,on='file_name')

        training_df.sort_values(by='file_name')
        # print(training_df)

        training_data_numpy_array = numpy.array(training_df.values)

        return training_data_numpy_array
        #
        #
        # print(numpy.array(training_df.values).shape)

        # (image_features_rows, count_feature_rows) = self.load_image_features('./../csv/input-file.txt')
        # (correct_outputs_rows, count_correct_outputs_rows) = self.load_correct_outputs('./../csv/correct-outputs.txt')
        #
        #
        # training_data = numpy.zeros((count_feature_rows,4))
        # i = 0
        # for row_features in image_features_rows:
        #     file_name = row_features[0]
        #     feature1 = row_features[1]
        #     feature2 = row_features[2]
        #     feature3 = row_features[3]
        #
        #     print(row_features)
        #
        #     training_data[i] = [float(feature1), float(feature2), float(feature3),float(0)]
        #
        #     for row_correct_outputs in correct_outputs_rows:
        #         if file_name == row_correct_outputs[0]:
        #             training_data[i][4] = row_correct_outputs[1]
        #             break
        #
        #     i = i+1
        #
        #
        # return training_data