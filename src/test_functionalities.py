''' This is where we will conduct the unit tests will be done using pytest module'''

import pytest
import pandas as pd
import pickle
from config import train_set, test_set
from preprocess import DataPreprocessing

class TestFunctions:

    def test_dataset_type(self):
        '''
        Unit test to see if the load function returns a dataframe
        :return: True is dataset is a DataFrame
        '''
        dp_obj = DataPreprocessing()
        dataset = dp_obj.load_data(train_set)
        assert isinstance(dataset, pd.DataFrame) == True
