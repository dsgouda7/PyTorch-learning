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
X_list = cars.wt.values
X_np = np.array(X_list, dtype=np.float32).reshape(-1,1)
y_list = cars.mpg.values
y_np = np.array(y_list, dtype=np.float32).reshape(-1,1)
X = torch.from_numpy(X_np)
y_true = torch.from_numpy(y_np)

#%% model class
# every Neural Network, layer or custom model in pytorch
# must inherit from torch.nn.Module
class LinearRegressionTorch(nn.Module):
    def __init__(self, input_size, output_size):
        # python convention to init the base class
        super(LinearRegressionTorch, self).__init__()
        # create 1 single linear regression layer of the form
        # y = wx+b
        self.linear = nn.Linear(input_size, output_size)

    def forward(self, x):
        # forward prop step where we
        # generate predictions using current parameters in the model
        return self.linear(x)

# input and output are of the (n,1) shape
input_dim = 1
output_dim = 1
model = LinearRegressionTorch(input_dim, output_dim)

# %% Init the loss function
loss_func = nn.MSELoss()

# %% init the optimizer
lr = 0.002
# optimizer is how the next step is taken
# in techniques like AdaGrad or Adam, the learning rate decays as the 
# loss starts approaching the minima
# the SGD technique refers to Stochastic gradient descent
# where the next step is calculated by sampling the data instead
# of calculating the gradient over the entire data set
# (that would be Batch Gradient descent)
optimizer = torch.optim.SGD(model.parameters(), lr=lr)

# %% Train the model
losses, slope, bias = [], [], []

num_epochs = 10000
for epoch in range(num_epochs):
    # init gradients to zero
    optimizer.zero_grad()

    # forward pass
    y_pred = model(X)

    # compute the loss
    loss = loss_func(y_pred, y_true)
    # this single line figures out the gradients using the derivative 
    # of the original function and updates the params
    loss.backward()

    # update the weights
    optimizer.step()

    # get slope and weights 
    for name, param in model.named_parameters():
        if param.requires_grad:
            if name == 'linear.weight':
                slope.append(param.data.numpy()[0][0])
            if name == 'linear.bias':
                bias.append(param.data.numpy()[0])

    # store the losses
    losses.append(float(loss.data))

    if(epoch%100==0):
        print('Epoch: {}, Loss: {:4f}'.format(epoch, loss.data))

# %% visualize the error
sns.lineplot(x=range(num_epochs), y=losses)

# %% visualize the movement of the slope/bias values
sns.scatterplot(x=range(num_epochs), y=slope)
sns.scatterplot(x=range(num_epochs), y=bias)

# %% check the result
y_pred = model(X).data.numpy().reshape(-1)
sns.scatterplot(x=X_list, y=y_list)
sns.lineplot(x=X_list, y=y_pred, color='red')

# %%
