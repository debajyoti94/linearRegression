""" In this .py file we will create functions for training
the model and retrieving the loss value
Training curves are plotted here"""

from models import LinearRegressionFromScratch
import argparse
from matplotlib import pyplot as plt
from preprocess import DataPreprocessing
from config import train_set, test_set


def make_loss_plots(X, y, lr_obj):
    '''

    :param X: input features
    :param y: ouput wine quality
    :param lr_obj: linear regression object for calling the
    function of gradient descent
    :return: model parameters (theta)
    '''

    # tried these following learning rates
    # learning_rates = [0.05, 0.005, 0.0005]
    learning_rates = [0.05]
    print(type(y))
    for lr in learning_rates:
        theta, cost_history = lr_obj.lr_gradient_descent(X, y, 0, lr)
        plt.plot(cost_history, linewidth=2)
    plt.title("Gradient descent with different Learning Rates", fontsize=16)
    plt.xlabel("Number of iterations", fontsize=14)
    plt.ylabel("Cost", fontsize=14)
    plt.grid()
    plt.legend(list(map(str, learning_rates)))
    plt.savefig("../plots/Different_learning_rates_gradient_descent.png")

    return theta


if __name__ == "__main__":
    # print("here")
    # first we get the data
    dp_obj = DataPreprocessing()
    # print(dir(dp_obj))
    raw_data = dp_obj.load_data(train_set)
    scaled_data = dp_obj.feature_scaling_data(raw_data)

    X_train = scaled_data.loc[:, scaled_data.columns != dp_obj.output_feature]
    y_train = scaled_data[dp_obj.output_feature]
    # print(type(y_train))

    # adding commandline arguments
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--train", type=str,
        help="Provide an option for training the model. OLS/GD"
    )
    parser.add_argument(
        "--test", type=str,
        help="Parameters of OLS/GD? Provide an option"
    )

    args = parser.parse_args()
    # print(args)
    lr_obj = LinearRegressionFromScratch()

    if args.train == "OLS":
        # next pass the data to the model
        print("OLS")
        # print(type(y_train))
        model_parameters_normal_equation = lr_obj.lr_normal_equation(X_train,
                                                                     y_train)

        # pickling the model parameters
        dp_obj.pickle_dump_model(model_parameters_normal_equation, 'OLS')

    elif args.train == "GD":
        # then we make the plots
        # print(type(y_train))
        print("GD")
        model_parameters_GD = make_loss_plots(X_train, y_train, lr_obj)

        dp_obj.pickle_dump_model(model_parameters_GD, 'GD')

    elif args.test == "OLS":
        print("Model evaluation using parameters from OLS")
        # firstly load the parameters
        theta = dp_obj.load_pickled_file('OLS')

        # load the test set
        wine_test_set = dp_obj.load_data(test_set)

        # here obtain the loss value
        X_test = wine_test_set.loc[:, wine_test_set.columns != dp_obj.output_feature]
        y_test = wine_test_set[dp_obj.output_feature]

        MSE_test = lr_obj.cost_function(theta=theta,
                                        wine_input_features=X_test,
                                        wine_quality_output=y_test)
        print("MSE loss using OLS = {}".format(MSE_test))

    elif args.test == "GD":
        print("Model evaluation using parameters obtained from GD")
        # load the parameters
        theta = dp_obj.load_pickled_file('GD')

        # once the parameters are obtained, next we have to load the test set
        wine_test_set = dp_obj.load_data(test_set)

        # here obtain the loss value
        X_test = wine_test_set.loc[:, wine_test_set.columns != dp_obj.output_feature]
        y_test = wine_test_set[dp_obj.output_feature]

        MSE_test = lr_obj.cost_function(theta=theta,
                                        wine_input_features=X_test,
                                        wine_quality_output=y_test)

        print("MSE loss using GD = {}".format(MSE_test))