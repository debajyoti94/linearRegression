''' In this .py file we will create functions for training the model and retrieving the loss value
Training curves are plotted here'''

from models import LinearRegressionFromScratch
import argparse
from matplotlib import pyplot as plt
from preprocess import DataPreprocessing
from config import train_set, test_set

def make_loss_plots(X, y, lr_obj):
    learning_rates = [0.05, 0.005, 0.0005]
    print(type(y))
    for lr in learning_rates:
        _, cost_history = lr_obj.lr_gradient_descent(X, y, 0, lr)
        plt.plot(cost_history, linewidth=2)
    plt.title("Gradient descent with different Learning Rates", fontsize=16)
    plt.xlabel("Number of iterations", fontsize=14)
    plt.ylabel("Cost", fontsize=14)
    plt.grid()
    plt.legend(list(map(str, learning_rates)))
    plt.savefig('../plots/Different_learning_rates_gradient_descent.png')



if __name__ == '__main__':
    # print("here")
    # first we get the data
    dp_obj = DataPreprocessing()
    raw_data = dp_obj.load_data(train_set)
    scaled_data = dp_obj.feature_scaling_data(raw_data)

    X_train = scaled_data.loc[:,scaled_data.columns != dp_obj.output_feature]
    y_train = scaled_data[dp_obj.output_feature]
    # print(type(y_train))

    # adding commandline arguments
    parser = argparse.ArgumentParser()

    parser.add_argument("--option", type=str, help="Provide an option for training the model. OLS/GD")

    args = parser.parse_args()
    # print(args)
    lr_obj = LinearRegressionFromScratch()

    if args.option == 'OLS':
        # next pass the data to the model
        print("OLS")
        # print(type(y_train))
        model_parameters_normal_equation = lr_obj.lr_normal_equation(X_train, y_train)
        print(model_parameters_normal_equation)
    elif args.option == 'GD':
        # then we make the plots
        # print(type(y_train))
        print("GD")
        make_loss_plots(X_train, y_train, lr_obj)

