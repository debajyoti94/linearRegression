""" This file will contain the code for implementing linear regression model using
- Gradient descent technique"""


from config import num_of_epochs
import numpy as np


class LinearRegressionFromScratch:
    def __init__(self):
        self.num_of_epochs = num_of_epochs

    def lr_normal_equation(self, wine_input_features, wine_quality_output):
        """
        theta = (X^T*X)^-1*X*y, this is the normal equation
        :param wine_input_features: the input features
        :param wine_quality_ouput:  the wine quality-- output
        :return: theta-- set of parameters
        """
        # converting the dataframe to numpy array
        X = wine_input_features.to_numpy()
        print("Inside normal equation function:{}".format(type(wine_quality_output)))
        y = wine_quality_output.to_numpy()
        print("Inside normal equation function:{}".format(type(y)))

        # applying the normal equation
        # below @ is used for matrix multiplication
        theta = np.linalg.inv(X.T @ X) @ X.T @ y
        print(theta.shape)
        return theta

    def cost_function(self, theta, wine_input_features, wine_quality_output):
        """
        Calculating MSE
        :param theta: model parameters
        :param wine_input_features: input
        :param wine_quality_output: output/predictions
        :return:
        """
        predictions = wine_input_features @ theta
        squared_errors = np.square(predictions - wine_quality_output)

        return np.sum(squared_errors) / (2 * len(wine_quality_output))

    def lr_gradient_descent(self, wine_input_features,
                            wine_quality_output, theta, lr):
        """

        :param wine_input_features: X
        :param wine_quality_output: y
        :param theta: model parameters
        :param lr: learning rate
        :return: model parameters(theta), loss values
        """
        # print(wine_input_features.shape)
        # print(wine_quality_output.shape)
        X = wine_input_features.to_numpy()
        y = wine_quality_output.to_numpy()

        cost_history = np.zeros(
            self.num_of_epochs
        )  # create a vector to store the cost history
        m = y.size  # number of training examples
        theta = np.zeros(wine_input_features.shape[1])

        for i in range(self.num_of_epochs):
            print("Epoch {}...".format(i))
            predictions = np.dot(X, theta)
            theta = theta - lr * (1.0 / m) * np.dot(X.T, predictions - y)
            cost_history[i] = self.cost_function(
                theta, X, y
            )  # compute and record the cost

        return theta, cost_history

