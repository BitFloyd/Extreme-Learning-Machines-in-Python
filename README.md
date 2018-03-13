# Extreme-Learning-Machines-in-Python

This is a simple ELM implementation in python

Dataset
--------
SKLEARN DIGITS DATASET.

Implementation:
--------------
The ELM class is used to create and evaluate an ELM (random activations with linear transformation coefficients sampled from a uniform distribution). 

The ELM class can be tuned to have any number of layers and hidden units and the group from which the activations are sampled can also be chosen.

The transformed data is used to train an SVM and test the accuracy against the SVM on raw data and a simple ANN.

ELMs using different number of hidden units are trained and plotted against their accuracies.

