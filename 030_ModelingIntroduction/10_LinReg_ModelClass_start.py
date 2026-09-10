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

# %% How the model, loss, and optimizer are connected
# These objects are not all directly linked to each other.
#
# model.parameters() -- same Parameter objects --> optimizer.param_groups
#        |
#        | model(X): autograd records operations that use the parameters
#        v
#   y_pred -- loss_func(y_pred, y_true) --> loss tensor
#                                                |
#                                          loss.backward()
#                                                v
#                                  model parameter .grad values
#                                                |
#                                          optimizer.step()
#                                                v
#                                   model parameter values change
#
# The loss function does not contain the model or optimizer. The connection is
# created dynamically because y_pred was calculated from model parameters.
#
# PYTORCH SHORTCOMING: PARAMETER OWNERSHIP IS IMPLICIT
# ---------------------------------------------------
# The model registers the parameters, but it is not their exclusive owner:
#
# - autograd writes each parameter's .grad state when loss.backward() runs;
# - optimizer.zero_grad() clears that .grad state;
# - optimizer.step() mutates the parameter values;
# - user code can also mutate either value inside torch.no_grad().
#
# Strictly speaking, the loss function does not mutate the parameters. The loss
# tensor asks autograd to do so through backward(). Still, these components
# coordinate through shared mutable Parameter objects instead of explicit inputs
# and return values. That makes ownership and data flow harder to see and allows
# mistakes such as stale gradients, optimizing the wrong parameters, or attaching
# multiple optimizers to the same parameters without an obvious warning.
model_parameters = list(model.parameters())
optimizer_parameters = [
    parameter
    for parameter_group in optimizer.param_groups
    for parameter in parameter_group['params']
]

optimizer_uses_model_parameters = all(
    model_parameter is optimizer_parameter
    for model_parameter, optimizer_parameter in zip(
        model_parameters, optimizer_parameters
    )
)

assert optimizer_uses_model_parameters
assert len(list(loss_func.parameters())) == 0
print(f'Optimizer references model parameters: {optimizer_uses_model_parameters}')
print(f'Loss function trainable parameters: {len(list(loss_func.parameters()))}')

# %% Train the model
losses, slope, bias = [], [], []

num_epochs = 10000
for epoch in range(num_epochs):
    # The optimizer clears .grad on the model parameters it references.
    optimizer.zero_grad()

    # The forward pass creates an autograd graph from parameters to predictions.
    y_pred = model(X)

    # The loss extends that graph from predictions to one scalar value.
    loss = loss_func(y_pred, y_true)

    if epoch == 0:
        print(f'Prediction tracks gradients: {y_pred.requires_grad}')
        print(f'Loss autograd node: {type(loss.grad_fn).__name__}')

    # backward() traverses the graph and writes gradients to each parameter's
    # .grad attribute. It does not update the parameter values.
    loss.backward()

    if epoch == 0:
        first_gradients = {
            name: parameter.grad.detach().clone()
            for name, parameter in model.named_parameters()
        }
        print(f'Model gradients after backward: {first_gradients}')

    # step() reads those .grad values and updates the same Parameter objects.
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
