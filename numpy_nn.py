import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import torch

# NUMPY Neural Network
# H is the hidden dimension
N = 64 # batch size
D_in = 1000 # dimensions in
H = 100 # hidden dimension
D_out = 10 # dims out

# # x is input data - numpy n x d array
# # n = 5, d = 100
# x = tn_feat_np
# # y is output data
# y = tn_trgt_np

x = np.random.randn(N, D_in)
y = np.random.randn(N, D_out)

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
