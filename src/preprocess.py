''' in this code we will pre-process the input data.
-basically apply feature scaling-minmax scaler
-no need to apply data imputation as there are no missing values
-also no categorical features were detected during EDA, hence not applying any encoding'''

import pandas as pd                             #PANDAS for reading and analyzing the input files
import pickle                                   #FOR OBJECT SERIALIZATION
from sklearn.preprocessing import MinMaxScaler  #THIS IS FOR FEATURE SCALING OPS

class DataPreprocessing(object):

    def __init__(self):
        '''
        constructor, setting up the minmax scaler here
         and also setting up the feature name we do not want to scale
        '''
        self.scaler = MinMaxScaler()
        self.output_feature = 'quality'

    def load_data(self, filename):
        '''
        Function to load the pickled dataset
        :param filename: file that we want to load
        :return: loaded dataset
        '''
        with open(filename, 'rb') as pickle_handle:
            dataset = pickle.load(pickle_handle)

        return dataset

    def feature_scaling_data(self, dataset):
        '''
        Function to apply minmax scaling to a dataset
        :param dataset: the dataset to which we want to apply scaling
        :return: scaled dataset
        '''
        columns_ = dataset.columns.drop(self.output_feature)
        dataset_scaled = dataset
        dataset_scaled[columns_] = self.scaler.fit_transform(dataset[columns_])

        return dataset_scaled