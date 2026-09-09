#%% packages
import numpy as np
import pandas as pd
import torch
import torch.nn as nn 
import seaborn as sns

#%% data import
cars_file = 'https://gist.githubusercontent.com/noamross/e5d3e859aa0c794be10b/raw/b999fb4425b54c63cab088c0ce2c0d6ce961a563/cars.csv'
cars = pd.read_csv(cars_file)
cars.head()

#%% visualise the model
sns.scatterplot(x='wt', y='mpg', data=cars)
sns.regplot(x='wt', y='mpg', data=cars)

#%% convert data to tensor
# capture the wt values
X_list = cars.wt.values
# however these raw values will cause exploding gradients
# we MUST scale this
X_list = X_list/X_list.max()

# when we read raw data from a csv the shape is usually (n,), however, most
# py ML libs work best when the shape is (n,1) instead
# -1 here says collapse all the existing dimensions of the np array into one single dimension
# for example (n,) => (n,1)
# (n,m,) => (n*m,1) and so on
X_np = np.array(X_list, dtype=np.float32).reshape(-1,1)
X_np.shape
Y_list = cars.mpg.values
Y_np = np.array(Y_list, dtype=np.float32).reshape(-1,1)

# 2 different ways to create a tensor, one from an np array and one from a py list
X = torch.from_numpy(X_np)
Y = torch.from_numpy(Y_np)
print(type(X))
print(type(Y))

# convert the np arrays to tensors, tensors are nothing but floating point type arrays with 
# in built functionalities like grad() to calculate gradients internally

# now populate random initial weights in a tensor
# requires_grad=True indicates we are going to calculate dw (gradients) for this tensor later
w = torch.rand((1,1), requires_grad=True, dtype=torch.float32)
b = torch.rand(1, requires_grad=True, dtype=torch.float32)

#%% training
# init the hyperparams
num_epochs = 10000
learning_rate = 0.01

#%% check results
for epoch in range(num_epochs):
    # forward prop
    y_pred = torch.matmul(X, w) + b

    # calculate the ms loss for the entire set of examples for this epoch
    loss = torch.mean((y_pred - Y)**2)

    # backprop => this single line of code calculates the dLoss/dw and dLoss/db
    # and stores it in the w.grad and b.grad properties
    loss.backward()

    # Update the weights with the gradients calculated
    with torch.no_grad():
        w -= (learning_rate*w.grad)
        b -= (learning_rate*b.grad)

        # once we are done calculating we need to set the grad to 0s again
        # if we don't do that the gradients that get calculated subsequently 
        # are subtracted/added to the existing values of the w.grad/b.grad values
        # which will break the whole gradient update logic
        w.grad.zero_()
        b.grad.zero_()

        print(loss)

# %%
# print what gradient descent got us
# Note: we used exactly one weight and bias in this example
print(f"weight: {w.item()} bias: {b.item()}")

# %% (Statistical) Linear Regression
# the result of X*w+b would be another tensor which have the pytorch
# engine tacked on them by default
# we need to detach the pytorch engine and convert it to a standard 
# np array in order to pass it to
# utility libs like sns and matplotlib
y_pred = (torch.matmul(X,w) + b).detach().numpy()

# %% create graph visualisation

sns.scatterplot(x=cars.wt.values, y=Y_list)
sns.lineplot(x=cars.wt.values, y=y_pred.squeeze())

# make sure GraphViz is installed (https://graphviz.org/download/)
# if not computer restarted, append directly to PATH variable
# %%
# compare the tensors with a bog standard sklearn linear regression model
from sklearn.linear_model import LinearRegression
reg = LinearRegression().fit(X_np, Y_list)
print(f'Slope: {reg.coef_}, Intercept: {reg.intercept_}')
# %%
