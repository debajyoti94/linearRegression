''' In this .py file we will create functions for training the model and retrieving the loss value
Training curves are plotted here'''

from models import LinearRegressionFromScratch
import argparse
from matplotlib import pyplot as plt
from preprocess import DataPreprocessing
from config import train_set, test_set

def make_loss_plots(X, y):
    learning_rates = [0.000008, 0.000005, 0.00001]
    for lr in learning_rates:
        _, cost_history = LinearRegressionFromScratch.lr_gradient_descent(X, y, 0, lr)
        plt.plot(cost_history, linewidth=2)
    plt.title("Gradient descent with different learning rates", fontsize=16)
    plt.xlabel("number of iterations", fontsize=14)
    plt.ylabel("cost", fontsize=14)
    plt.legend(list(map(str, learning_rates)))


if __name__ == '__main__':
    # print("here")
    # first we get the data
    dp_obj = DataPreprocessing()
    raw_data = dp_obj.load_data(train_set)
    scaled_data = dp_obj.feature_scaling_data(raw_data)

    X_train = scaled_data.loc[:,scaled_data.columns != dp_obj.output_feature]
    y_train = scaled_data[dp_obj.output_feature]

    # adding commandline arguments
    # parser = argparse.ArgumentParser()
    #
    # parser.add_argument("OLS")
    # parser.add_argument("GD")
    #
    # args = parser.parse_args()
    # print(args)
    lr_obj = LinearRegressionFromScratch()

    # if args.OLS:
        # next pass the data to the model
    print("OLS")
    model_parameters_normal_equation = lr_obj.lr_normal_equation(X_train, y_train)
    print(model_parameters_normal_equation)
    # elif args.GD:
        # then we make the plots
        # make_loss_plots(X_train, y_train)

