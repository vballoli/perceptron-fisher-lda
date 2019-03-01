from utils.utility import log
import numpy as np
import pickle

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

    def predict(self, X, weights=None, info=None):
        """
        Single epoch training of the perceptron based on the outputs.
        """
        log(should_print=self.log, "Predicting: ", info)
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
                prediction = self.predict(X, y, weights, info="Epoch: " + epoch)
                improv = (y - prediction) * learning_rate
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

    def predict(self, X_test):
        """
        Predicts the labels of the test data based on the trained model.
        """
        pass

    def evaluate(self, y_predicted, y_test):
        """
        Evaluates the performance of the trained perceptron model based on test data.
        """
        pass

