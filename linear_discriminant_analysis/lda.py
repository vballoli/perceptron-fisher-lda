import numpy as np
from numpy.linalg import eig, inv
#from utils.utility import log

class LinearDiscrimnantAnalysis:
    """
    Implements Fisher's Linear Discriminant Analysis for 2 classes only.
    
    Pseudo Algorithm:
    1. Find individual means for each class data [mew1, mew2]
    2. Find covariance matrix of each class data [S1, S2]
    3. Within class scatter matrix = sum(all covariane matrices) [Sw = S1+S2]
    4. Between class scatter matrix [Sb = (mew2 - mew1) * (mew2 - mew1).T]
    5. w = (Sw^-1) * (mew2 - mew1)

    """

    def __init__(self, log):
        """
        Initailizes class parameters.
        """
        self.log = log

    def fit(self, df, target):
        """
        Computes the weight vector based on the training data and corresponding class labels.
        """
        mean = np.array(df.groupby(target).mean()).T
        mew1, mew2 = mean[0], mean[1]
        classes = df[target].unique()
        X = df[df.columns.difference([target])]
        groups = []
        for label in classes:
            groups.append(df.groupby(target).groups.get_group(label))
        groups_covariance = []
        for group in groups:
            groups_covariance.append(np.cov(group))
        Sw = sum(groups_covariance)
        w = inv(Sw) * (mew2 - mew1)
        self.w = w
        self.m = np.array(df[df.columns.difference([target])].mean()).T[0]
        self.classes = df.columns.difference([target])
        return w, X



    def predict(self, w, m, X):
        """
        Predicts the class label based on trained data.
        """
        if self.w is not None:
            w = self.w
        if self.m is not None:
            m = self.m
        prediction = w * (X - m)
        if prediction > 0:
            return self.classes[0]
        else:
            return self.classes[1]
        