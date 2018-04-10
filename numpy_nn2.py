import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import torch

# NUMPY Neural Network
# H is the hidden dimension
N = 10 # batch size
D_in = 5 # dimensions in
H = 20 # hidden dimension
D_out = 6 # dims out

# x is input data: (loan_amnt, annual_inc)
# y is output target: (int_rate)
X = np.array(([-0.89398298, -1.24346915], [-1.30913251, -1.02124544], [-1.32573849, -1.67858318]), dtype=float)
Y = np.array(([0.59916467], [0.56863012], [0.74304103]), dtype=float)

# x = np.random.randn(N, D_in)
# y = np.random.randn(N, D_out)

# randomly initialise weights
# 5 predictors and 6 output target classes
# w1 = np.random.randn(100, H)
# w1 = np.random.randn(5, 100)
# w2 = np.random.randn(100, 6)

w1 = np.random.randn(D_in, H)
w2 = np.random.randn(H, D_out)

# learning rate - can modify this to improve accuracy
learning_rate = 0.001

for t in range(500):
    # Forward pass: compute predicted y
    h = x.dot(w1)
    h_relu = np.maximum(h,0)
    y_pred = h_relu.dot(w2)

    # Compute and print loss (using RSME)
    (y_pred - y).sum()
    loss = np.square(y_pred - y).sum()
    print(t, loss)

    # Backpropogate to compute gradients of w1 and w2 with respect to loss
    grad_y_pred = 2.0 * (y_pred - y)
    grad_w2 = h_relu.T.dot(grad_y_pred)
    grad_h_relu = grad_y_pred.dot(w2.T)
    grad_h = grad_h_relu.copy()
    grad_h[h<0] = 0
    grad_w1 = x.T.dot(grad_h)

    # Update weights
    w1 -= learning_rate * grad_w1
    w2 -= learning_rate * grad_w2
