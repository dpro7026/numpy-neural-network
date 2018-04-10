# clean up data and store into train and test csv


# Import cleaned data from csv
# Use 10 nodes in hidden layer
# Switch to RelU activation function
# Batching??

import numpy as np
from numpy import genfromtxt

# 1000x3 matrix
X = genfromtxt('training_data.csv', delimiter=',')
X = np.delete(X, 0, 0) # delete headers
X = np.delete(X, 0, 1) # delete the id column
print(X)

# 1000x7 matrix
Y = genfromtxt('encoded_training_target.csv', delimiter=',')
Y = np.delete(Y, 0, 0) # delete headers
Y = np.delete(Y, 0, 1) # delete the id column
print(Y)

# define neural network
class Neural_Network(object):
    def __init__(self):
        # parameters
        self.input_size = 3
        self.output_size = 7
        self.hidden_size = 10

        # generate weights randomly
        self.W1 = np.random.randn(self.input_size, self.hidden_size)
        self.W2 = np.random.randn(self.hidden_size, self.output_size)

    def forward(self, X):
        # forward propogation through the network
        self.z = np.dot(X, self.W1) # dot product of X (input) and first set of 3x2 weights
        self.z2 = self.sigmoid(self.z) # activation function
        self.z3 = np.dot(self.z2, self.W2) # dot product of hidden layer and second set of 3x1 weights
        o = self.sigmoid(self.z3) # final activation function
        return o

    def sigmoid(self, s):
        # activation function
        return 1 / (1 + np.exp(-s))

    def sigmoidPrime(self, s):
        # derivative of sigmoid
        return s * (1 - s)

    def backward(self, X, Y, o):
        # backward propogation through the network
        self.o_error = Y - o # error in output
        self.o_delta = self.o_error * self.sigmoidPrime(o) # apply sigmoid prime to error

        self.z2_error = self.o_delta.dot(self.W2.T) # how much our hidden layer weights contributed to error
        self.z2_delta = self.z2_error * self.sigmoidPrime(self.z2) # apply sigmoid to z2 error

        self.W1 += X.T.dot(self.z2_delta) # adjust first weights
        self.W2 += self.z2.T.dot(self.o_delta) # adjust second set of weights

    def train (self, X, Y):
        o = self.forward(X)
        self.backward(X, Y, o)


NN = Neural_Network()
for i in range(1000): # train the NN 1,000 times
    # correct = 0
    print(f'Input: {str(X)}')
    print(f'Actual Output: {str(Y)}')
    print(f'Predicted Output: {str(NN.forward(X))}')
    print(f'Loss: {str(np.mean(np.square(Y - NN.forward(X))))}') # mean sum squared loss

    NN.train(X, Y)
np.savetxt('predicted.csv', NN.forward(X), delimiter=',')
