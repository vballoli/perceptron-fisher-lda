import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import seaborn as sns

class Perceptron:
    """
    Implements the perceptron algorithm.
    """

    def __init__(self, learning_rate, epochs, input_size, threshold, dataset_label, img_dir):
        """
        Initialize class parameters
        @param learning_rate: Learning rate is magnitude of step taken along the gradient direction
        @param epochs: Total epochs the data must be trained on.
        @param input_size: Number of features for input data.
        @param threshold: Threshold for the activation function ot classify as 1 or -1.
        @param dataset_label: String to label the given trainig data
        @param img_dir: Destination path for the images of the plots of the perceptron algorithm.
        """
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.W = np.ones(input_size+1) # +1 for bias
        self.threshold = threshold
        self.dataset_label = dataset_label
        self.img_dir = img_dir
        self.trained = False

    def activation(self, x):
        """
        Activation function defined: 1 if x > threshold else -1.
        @param x: activation function input
        """
        if x >= self.threshold:
            prediction = 1
        else:
            prediction = -1
        return prediction

    def fit(self, X, y):
        """
        Train the perceptron based on input training data and class labels.
        @param X: training feature data.
        @param y: class labels for the given training data.
        """
        w = self.W
        self.X = X
        self.y = y
        self.visualize_data(None, self.dataset_label, 0)
        X = np.insert(X, 0, 1, axis=1)
        for epoch in range(self.epochs):
            not_equal = 0
            for i in range(X.shape[0]):
                if self.activation(np.dot(w, X[i])) != y[i]:
                    not_equal += 1
                    w = w + self.learning_rate * X[i] * y[i]
            self.visualize_data(w, self.dataset_label, epoch+1)
        self.w = w
        self.trained = True

    def predict(self, X):
        """
        Predict the class label for input test data
        @param X: test data.
        """
        if self.trained:
            try:
                return self.activation(np.dot(self.w, X))
            except Exception as e:
                print("Error:", '\t', e)
                return None
        else:
            print("Untrained pereceptron. No prediction")
            return None



    def visualize_data(self, w, name=None, epoch=None):
        """
        Visualize the perceptron algorithm for given weight vector and training data.
        @param w: Trained weight vector from teh dataset.
        @param name: Name of the plot.
        @param epoch: Epoch iteration.
        """
        X = self.X
        if w is not None:
            x = np.array([-3.8, 3])
            y = (w[0] + w[1] * x) / w[2] * -1
            plt.plot(x, y, c='green')
        y = self.y
        colour = ['red' if i== 1 else 'blue' for i in y]
        plt.scatter(X.T[0], X.T[1], c=colour)
        plt.suptitle('Dataset ' + name + ' Epoch: ' + str(epoch))
        plt.title('Learning rate: ' + str(self.learning_rate))
        plt.savefig(self.img_dir+'Dataset_' + self.dataset_label + "_Epoch_" + str(epoch) +'.png')

