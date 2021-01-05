""" In config.py we can setup the hyper-parameters of the model and of the script in general
This file will contain the variables which are useful for training the model."""

model_name_OLS = "../models/LR_OLS.pickle"
model_name_GD = "../models/LR_GD.pickle"

num_of_epochs = 1000  # USED in gradient descent

train_set = "../input/wine_train_set.pickle"
test_set = "../input/wine_test_set.pickle"
