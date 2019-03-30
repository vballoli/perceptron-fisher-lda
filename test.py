from perceptron.perceptron import Perceptron
from linear_discriminant_analysis.fisher_lda import FisherLDA
import pandas as pd
import numpy as np

"""
df = pd.read_csv('datasets/dataset_3.csv', names=['X1', 'X2', 'y'])
df['y'] = df['y'].replace(0, -1)
print(df.head())
p = Perceptron(0.01, 20, 2, 0, '3', './images/')
X = np.array(df[['X1', 'X2']])
y = df['y']
p.fit(X, y)
"""
disc = FisherLDA(1)
disc.visualize()
