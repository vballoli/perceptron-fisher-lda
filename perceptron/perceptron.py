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

    def __init__(self, learning_rate, epochs, input_size, threshold, dataset_label, img_dir):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.W = np.ones(input_size+1) # +1 for bias
        self.threshold = threshold
        self.dataset_label = dataset_label
        self.img_dir = img_dir

    def activation(self, x):
        if x >= self.threshold:
            prediction = 1
        else:
            prediction = -1
        return prediction

    def fit(self, X, y):
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

    def visualize(self):
        w = self.w
        X = self.X
        x = np.random.random_sample(2)
        y = (w[0] + (w[1] * x)) / w[2]
        plt.plot(x, y)
        plt.show()

    def visualize_data(self, w, name=None, epoch=None):
        X = self.X
        #x = np.random.random_sample(100)
        if w is not None:
            x = np.array([-3.8, 3])
            y = (w[0] + w[1] * x) / w[2] * -1
            plt.plot(x, y, c='green')
        #plt.scatter('X1', 'X2', data=pd.read_csv(path, names=['X1', 'X2', 'y']), hue='y', fit_reg=False)
        y = self.y
        colour = ['red' if i== 1 else 'blue' for i in y]
        plt.scatter(X.T[0], X.T[1], c=colour)
        plt.suptitle('Dataset ' + name + ' Epoch: ' + str(epoch))
        plt.title('Learning rate: ' + str(self.learning_rate))
        plt.savefig(self.img_dir+'Dataset_' + self.dataset_label + "_Epoch_" + str(epoch) +'.png')

