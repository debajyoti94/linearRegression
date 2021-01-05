""" in this code we will pre-process the input data.
-basically apply feature scaling-minmax scaler
-no need to apply data imputation as there are no missing values
-also no categorical features were detected during EDA, hence not applying any encoding"""

import pandas as pd  # PANDAS for reading and analyzing the input files
import pickle  # FOR OBJECT SERIALIZATION
from sklearn.preprocessing import MinMaxScaler  # THIS IS FOR FEATURE SCALING OPS
from config import model_name_GD, model_name_OLS

class DataPreprocessing(object):
    def __init__(self):
        """
        constructor, setting up the minmax scaler here
         and also setting up the feature name we do not want to scale
        """
        self.scaler = MinMaxScaler()
        self.output_feature = "quality"
        self.model_name_GD = model_name_GD
        self.model_name_OLS = model_name_OLS

    def load_data(self, filename):
        """
        Function to load the pickled dataset
        :param filename: file that we want to load
        :return: loaded dataset
        """
        with open(filename, "rb") as pickle_handle:
            dataset = pickle.load(pickle_handle)

        return dataset

    def feature_scaling_data(self, dataset):
        """
        Function to apply minmax scaling to a dataset
        :param dataset: the dataset to which we want to apply scaling
        :return: scaled dataset
        """
        columns_ = dataset.columns.drop(self.output_feature)
        dataset_scaled = dataset
        dataset_scaled[columns_] = self.scaler.fit_transform(dataset[columns_])

        return dataset_scaled

    def pickle_dump_model(self, file, model_option):
        '''

        :param file:
        :param model_option:
        :return:
        '''
        if model_option == 'OLS':
            # do something here
            with open(self.model_name_OLS, 'wb') as pickle_handle:
                pickle.dump(file, pickle_handle)

        elif model_option == "GD":
            # do something here
            with open(self.model_name_GD, 'wb') as pickle_handle:
                pickle.dump(file, pickle_handle)

    def load_pickled_file(self, model_option):
        '''

        :param model_option:
        :return:
        '''
        if model_option == 'OLS':
            with open(self.model_name_OLS, 'rb') as pickle_handle:
                theta = pickle.load(pickle_handle)
        elif model_option == "GD":
            with open(self.model_name_GD, 'rb') as pickle_handle:
                theta = pickle.load(pickle_handle)

        return theta
