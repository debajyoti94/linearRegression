''' This file will contain the code for implementing linear regression model using
- Gradient descent technique'''


from config import num_of_epochs
import numpy as np

class LinearRegressionFromScratch:

    def __init__(self):
        self.num_of_epochs = num_of_epochs


    def lr_normal_equation(self, wine_input_features, wine_quality_ouput):
        '''
        theta = (X^T*X)^-1*X*y, this is the normal equation
        :param wine_input_features: the input features
        :param wine_quality_ouput:  the wine quality-- output
        :return: theta-- set of parameters
        '''
        # converting the dataframe to numpy array
        X = wine_input_features.to_numpy()
        y = wine_quality_ouput.to_numpy()

        #applying the normal equation below @ is used for matrix multiplication
        theta = np.linalg.inv(X.T@X)@X.T@y

        return theta


    def cost_function(self, theta, wine_input_features, wine_quality_output):
        '''
        Calculating MSE
        :param theta: model parameters
        :param wine_input_features: input
        :param wine_quality_output: output/predictions
        :return:
        '''
        predictions = wine_input_features @ theta
        squared_errors = np.square(predictions - wine_quality_output)

        return np.sum(squared_errors)/(2*len(wine_quality_output))

    def lr_gradient_descent(self, wine_input_features, wine_quality_output, lr):
        '''
        function for gradient descent
        :param wine_input_features:
        :param wine_quality_output:
        :param lr: learning rate
        :return:
        '''
        X = wine_input_features.to_numpy()
        y = wine_quality_output.to_numpy()

        cost_history = np.zeros(self.num_of_epochs)  # create a vector to store the cost history
        m = y.size  # number of training examples

        for i in range(self.num_of_epochs):
            predictions = np.dot(X, theta)
            theta = theta - lr * (1.0 / m) * np.dot(X.T, predictions - y)
            cost_history[i] = self.cost(theta, X, y)  # compute and record the cost

        return theta, cost_history


