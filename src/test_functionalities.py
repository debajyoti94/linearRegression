""" This is where we will conduct the unit tests will be done using pytest module"""


import pandas as pd
from config import train_set
from preprocess import DataPreprocessing


class TestFunctions:
    def test_dataset_type(self):
        """
        Unit test to see if the load function returns a dataframe
        :return: True is dataset is a DataFrame
        """
        dp_obj = DataPreprocessing()
        dataset = dp_obj.load_data(train_set)
        assert isinstance(dataset, pd.DataFrame) == True

    def test_feature_scaled_data(self):
        '''
        Unit test to check if the minmax scaler returns the dataset
        of the same shape
        :return: True if the shape of the dataset is the same as input
        '''
        dp_obj = DataPreprocessing()
        dataset = dp_obj.load_data(train_set)
        scaled_dataset = dp_obj.feature_scaling_data(dataset)
        if dataset.shape == scaled_dataset.shape:
            assert True