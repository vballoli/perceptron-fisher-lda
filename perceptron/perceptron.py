from utils.utility import log
import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import seaborn as sns

class Perceptron:
    """
    Implements the perceptron algorithm.
    """

    def __init__(self, log, learning_rate, epochs, seed, threshold):
        """
        Initializes class parameters.
        """
        self.log = log
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.seed = seed
        self.threshold = threshold

    def classify(self, X, weights=None, info=None):
        """
        Single epoch training of the perceptron based on the outputs.
        """
        activation = weights[1:] * X + weights[0]
        if activation >= self.threshold:
            prediction = 1
        else:
            prediction = 0
        return prediction


    def fit(self, X, y, weights=None, info=None):
        """
        Trains the perceptron based on the outputs.
        """
        print("Fitting data....", info)
        if weights is not None:
            weights = np.zeros(X.shape[0]+1)
        for epoch in range(self.epochs):
            for input_data, value in zip(X, y):
                prediction = self.predict(X, weights, info="Epoch: " + epoch)
                improv = (y - prediction) * self.learning_rate
                weights[1:] += improv * X
                weights[0] += improv
                self.save_model(weights, "weight_"+epoch)


    def save_model(self, weights, name):
        """
        Saves the trained perceptron model for further use.
        """
        if not str(name).endswith(".pickle"): name += ".pickle"
        with open(name, 'wb') as f:
            pickle.dumps(weights, f)

    def predict(self, X_test, weights, info=None):
        """
        Predicts the labels of the test data based on the trained model.
        """
        return 0

    def evaluate(self, y_predicted, y_test):
        """
        Evaluates the performance of the trained perceptron model based on test data.
        """
        pass


    def visualize(self, path):
        df = pd.read_csv(path, names=['x','y'])
        sns.lmplot('x', 'y', data=df, fit_reg=False)
        plt.title('Plot')
        plt.show()


