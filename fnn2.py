import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import torch

# import data from csv
df = pd.read_csv('loan.csv', nrows=110)
# print out list of all headers
headers = df.columns.values.tolist()
print(headers)
# print count for values for each 'grade'
print(df['grade'].value_counts())

# CLEAN DATA
# Choose a subset of predictors - target is 'grade' (of loan)
df = df[['loan_amnt', 'term', 'int_rate', 'annual_inc', 'grade']]
# get basic stats
print(df.describe(include='all'))
print(df.dtypes)
# Check what the categroies for 'term' are
print(df['term'].unique())

# Categorical to numerical values
# get the variable names for the dummy variable
dummy_1 = pd.get_dummies(df["term"])
print(dummy_1.head())
dummy_1.rename(columns={'term':'36 months', 'term':'60 months'}, inplace=True)
print(dummy_1.head())
# merge data frame "df" and "dummy_1"
df = pd.concat([df, dummy_1], axis=1)
# drop original column "fuel-type" from "df"
df.drop("term", axis = 1, inplace=True)
print(df.head())
print(df.dtypes)

# split dataframe into testing and training
df_train=df.copy(deep=True)
df_train=df_train.head(100)
df_test=df.copy(deep=True)
df_test=df_test.tail(10)

# Z-Score or Standard Score: new=old-mean/standard deviation
# Normalise ranges to within range (-3,3)
df_train['loan_amnt']=(df_train['loan_amnt']-df_train['loan_amnt'].mean())/df_train['loan_amnt'].std()
df_train['annual_inc']=(df_train['annual_inc']-df_train['annual_inc'].mean())/df_train['annual_inc'].std()
df_train['int_rate']=(df_train['int_rate']-df_train['int_rate'].mean())/df_train['int_rate'].std()
print(df_train.head())

df_test['loan_amnt']=(df_test['loan_amnt']-df_test['loan_amnt'].mean())/df_test['loan_amnt'].std()
df_test['annual_inc']=(df_test['annual_inc']-df_test['annual_inc'].mean())/df_test['annual_inc'].std()
df_test['int_rate']=(df_test['int_rate']-df_test['int_rate'].mean())/df_test['int_rate'].std()
print(df_test.head())

print(df_train.columns.values.tolist())
# training features
train_features = df_train[['loan_amnt', 'int_rate', 'annual_inc', ' 36 months', ' 60 months']]
# # training targets
train_targets = df_train[['grade']]
# testing features
test_features = df_test[['loan_amnt', 'int_rate', 'annual_inc', ' 36 months', ' 60 months']]
# testing targets
test_targets = df_test[['grade']]


# Construct a Neural Network
num_predictors = len(df_train.columns.values.tolist())

# dataframe to numpy array
tn_feat_np = train_features.values
tn_trgt_np = train_targets.values
print(tn_feat_np)

# numpy array to torch tensor
tn_feat_tnsr = torch.from_numpy(tn_feat_np)
print(tn_feat_tnsr)  # see how changing the np array changed the torch Tensor automatically



# NUMPY Neural Network
# H is the hidden dimension
H = 5
# x is input data - numpy n x d array
# n = 5, d = 100
x = tn_feat_np
# y is output data
y = tn_trgt_np

# randomly initialise weights
# 5 predictors and 6 output target classes
# w1 = np.random.randn(100, H)
w1 = np.random.randn(5, 100)
w2 = np.random.randn(100, 6)

# learning rate
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
