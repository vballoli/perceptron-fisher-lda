"""
Fisher's Linear Discriminant Analysis for 2 classes (with visualization)
Author: Rohan Tammara
Last Modified: 29/3/19
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats

class FisherLDA:

    def __init__(self, dataset):
        self.dataset = dataset

    def prepare_data(self):
        ### Prepare data ###
        data = pd.read_csv('datasets/dataset_' + str(self.dataset) + '.csv', header=None, prefix='D')
        data = data.iloc[:, 1:]
        X = np.array(data.iloc[:, :-1])
        # Datapoints belonging to class 0
        x1 = np.array(data.loc[lambda df: df.D3 == 0].iloc[:, :-1])
        # Datapoints belonging to class 1
        x2 = np.array(data.loc[lambda df: df.D3 == 1].iloc[:, :-1])
        # Labels
        y = np.array(data.iloc[:, 2])

        return x1, x2, y, X

    def fit(self, x1, x2, y):
        l1 = len(x1)
        l2 = len(x2)

        # Calculate mean values
        m1 = np.mean(x1, axis=0)
        m2 = np.mean(x2, axis=0)
        m = (l1*m1 + l2*m2)/(l1+l2)

        # Calculate within class variance matrix
        S1 = 0
        S2 = 0
        for i in range(l1):
            S1 += np.outer((x1[i]-m1), (x1[i]-m1).T)
        for i in range(l2):
            S2 += np.outer((x2[i]-m2), (x2[i]-m2).T)
        Sw = S1 + S2

        # Get the weight vector
        W = np.dot(np.linalg.inv(Sw), (m2 - m1))
        W = W/np.sqrt(W[0]**2 + W[1]**2)

        return W, m

    def plot(self, x1, x2, y, X, W, m):
        # Convenience variables
        L = len(X)
        L2 = np.count_nonzero(y)
        L1 = L - L2
        X = X.T
        W0sq = W[0]**2
        W1sq = W[1]**2

        # Discriminant line equation
        x_1 = np.linspace(-5, 5)
        x_0 = (-W[1]/W[0])*(x_1 - m[1]) + m[0]

        # Projection of X onto Discriminant (Transformed X)
        Xtr = np.zeros((2, L))
        for i in range(L):
            Xtr[0][i] = ((W1sq*m[1]+W0sq*X[0][i]) + ((m[0] - X[1][i])*W[0]*W[1]))/(W0sq+W1sq)
            Xtr[1][i] = (W[0]*(Xtr[0][i])/W[1]) + X[1][i]

        ### Normal Distribution Curves ###
        # Standard deviation interval size
        k = 3.7
        # Calculate points in lower dimension (here 1 and 2 are not features but classes)
        X1 = np.zeros(L1)
        X2 = np.zeros(L2)
        for i in range(L1):
            X1[i] = np.dot(W.T, x1[i])
        for i in range(L2):
            X2[i] = np.dot(W.T, x2[i])
        # Normal curve for X1
        mu1 = np.mean(X1)
        sigma1 = np.std(X1)
        p1 = np.linspace(mu1 - k*sigma1, mu1 + k*sigma1)
        q1 = stats.norm.pdf(p1, mu1, sigma1)
        # Normal curve for X2
        mu2 = np.mean(X2)
        sigma2 = np.std(X2)
        p2 = np.linspace(mu2 - k*sigma2, mu2 + k*sigma2)
        q2 = stats.norm.pdf(p2, mu2, sigma2)
        # Find the intersection point of distribution
        #intersection_pt =

        ### Plot ###
        grid = plt.GridSpec(3,2)
        plt.subplot(grid[0:2, :])
        clr = ['orange' if i == 0 else 'green' for i in y]
        clr_tr = ['red' if i == 0 else 'blue' for i in y]
        plt.scatter(X[0], X[1], marker='o', color=clr, alpha=0.75)
        plt.scatter(Xtr[0], Xtr[1], marker='.', color=clr_tr)
        plt.xlim(np.min(X[0])-0.1, np.max(X[0])+0.1)
        plt.ylim(np.min(X[1])-0.1, np.max(X[1])+0.1)
        plt.plot(x_1, x_0, color='k')
        plt.subplot(grid[2,:])
        plt.plot(p1, q1, color='red')
        plt.plot(p2, q2, color='blue')
        #plt.plot([intersection_pt], 'g^')
        plt.show()

    def visualize(self):
        x1, x2, y, X = self.prepare_data()
        W, m = self.fit(x1, x2, y)
        self.plot(x1, x2, y, X, W, m)
