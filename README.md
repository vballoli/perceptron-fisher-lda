# BITS F464 Machine Learning Assignment 1

This repository contains code and dataset for implementing Fisher's Linear Discriminant Analysis and the Perceptron Algorithm.

## Fisher's Linear Discriminant Analysis

This LDA algorithm uses Fisher's criterion to determine the weight vector and therefore the discriminant. We then project our points onto this single dimension and find the intersection of normal curves to find the optimum classification threshold thus minimizing the within-class variance and maximizing the between class difference.

`images/` folder contains the PNG files visualizing the discriminant, the sample's projection onto the discriminant and the normal distribution curves of the classes.

## Perceptron Algorithm

The perceptron algorithm implemented uses the activation function: f(x) = 1 if x >= threshold else -1, threshold usually set to zero.

`images/` folder contains GIFs of the change of decision boundary as the perceptron updates the weight vector based on the error of prediction.
