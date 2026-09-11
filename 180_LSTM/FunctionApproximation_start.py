#%% Packages
import numpy as np 
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import seaborn as sns
# %% simple dataset
num_points = 360*4

# x is a number between 0 and 360*4
X = np.arange(num_points)

# y is a noisy, amplitude-modulated cosine function of x
# the noise is Gaussian with standard deviation 0.1 introduced to the signal
# by adding a random Gaussian noise to each point

# np.cos(X[i]*np.pi/180) => generates the cosine wave for the given angle in degrees
# (1+i/num_points) =>  increases the extremities of the cosine wave linearly with i
# (np.random.randn()*0.1) => introduces the noise that ensures this is not a perfectly smooth signal
# All this to see if the model can learn the uber complex pattern in the data which in reality
# is just the cosine function with some amplitude modulation and noise
y = [np.cos(X[i]*np.pi/180) * (1+i/num_points) +
        (np.random.randn()*0.1) for i in range(len(X))]


sns.lineplot(x=X, y=y)


# %% Data Restructuring
# Build one input row from each sequence of 10 consecutive y-values.
# The value immediately after each sequence is its prediction target:
# X_restruct[0] = y[0:10]  -> y_restruct[0] = y[10]
# X_restruct[1] = y[1:11]  -> y_restruct[1] = y[11]
# In general: X_restruct[i] = y[i:i+10] -> y_restruct[i] = y[i+10]
# note: the i values used to calculate the cosine function are no longer relevant here
# the only input is the past sequence of 10 y-values
X_restruct = [] 
y_restruct = [] 

# iterate from i to num_points-10
for i in range(num_points-10):
    # append the collected sequence to X_restruct
    X_restruct.append(y[i:i+10])

    # append the next value as the target to y_restruct
    y_restruct.append(y[i+10])

# convert the lists to np arrays
X_restruct = np.array(X_restruct)
y_restruct = np.array(y_restruct)

# check the shapes of the restructured data arrays
# X_restruct should have shape (num_points-10, 10)
print(X_restruct.shape, y_restruct.shape, len(y))


# %% Train / Test Split
# keep 360*1 for testing and the rest for training
train_test_clipping = 360*3
X_train = X_restruct[:train_test_clipping]
X_test = X_restruct[train_test_clipping:]
y_train = y_restruct[:train_test_clipping]
y_test = y_restruct[train_test_clipping:]


#%% Create Dataset and Dataloader
# Dataset 
class TrigonometricDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# Dataloader
train_loader = DataLoader(TrigonometricDataset(X_train, y_train), batch_size=2)
test_loader = DataLoader(TrigonometricDataset(X_test, y_test), batch_size=len(y_test))

#%% visualize train data
sns.lineplot(x=range(len(y_train)), y=y_train, label = 'Train Data')

# %% Model
class TrigonometryModel(nn.Module):
    def __init__(self, input_size=1, output_size=1):
        super(TrigonometryModel, self).__init__()
        # Inside one LSTM time step (the same LSTM cell is reused for all 10 steps):
        #
        # The cell receives three things:
        #   1. The current time step's input value or feature values.
        #   2. The previous time step's 5-number hidden summary.
        #   3. The previous time step's 5-number long-term memory.
        #
        # It then performs these jobs:
        #   1. The forget gate decides how much of the old memory to keep.
        #   2. The candidate calculation proposes new information to remember.
        #   3. The input gate decides how much of that new information to save.
        #   4. The cell combines the kept old memory with the accepted new memory.
        #   5. The output gate decides which parts become the new hidden summary.
        #
        # Two 5-number values leave the cell:
        #   1. The updated long-term memory, passed to the next time step.
        #   2. The updated hidden summary, passed to the next time step and also
        #      recorded as the LSTM output for the current time step.
        #
        # After all 10 time steps, PyTorch has 10 hidden summaries for each sample.
        # forward() selects the last 5-number summary, and fc1 converts those five
        # numbers into one prediction. The returned status also contains the final
        # hidden summary and long-term memory, but this model does not use it.
        #
        # Model data flow for one training batch:
        #
        # DataLoader       reshape for LSTM       LSTM outputs          select last step       fc1 prediction
        #   (2, 10)   ->       (2, 10, 1)      ->   (2, 10, 5)    ->        (2, 5)         ->       (2, 1)
        # samples x values    batch x time x input   5 hidden values       final 5-value          one predicted
        #                                              per time step       sequence summary       value per sample
        #
        # In general:
        # (batch, 10, input_size) -> LSTM -> (batch, 10, hidden_size)
        # -> select the final time step -> (batch, hidden_size)
        # -> Linear(hidden_size, output_size) -> (batch, output_size)
        #
        # X_train has shape (1080, 10): 1080 samples, each containing 10 time steps.
        # X_test has shape (350, 10): 350 samples, each containing 10 time steps.
        # The training DataLoader returns (2, 10) because batch_size=2; before the
        # LSTM, it is reshaped to (2, 10, 1) = (batch, time steps, features).
        # The test batch similarly changes from (350, 10) to (350, 10, 1).
        # input_size is therefore 1: each time step contains one y-value feature.
        # With 2 features per step, the data shape would be (batch, 10, 2), and
        # input_size would be 2.
        # hidden_size=5 is not 5 layers; num_layers=1 means there is one LSTM layer.
        # At each of the 10 time steps, that layer updates a 5-value hidden state.
        # For an input shaped (batch, 10, x), the LSTM output is (batch, 10, 5):
        # each row of x features is processed in order into 5 learned state values.
        # In forward(), x[:, -1, :] selects the final 5-value sequence summary,
        # giving shape (batch, 5), which the fully connected layer maps to 1 value.
        #
        # hidden_size is a tunable model-capacity setting, not a count of layers.
        # Start small (for example 4, 8, 16, or 32) and compare validation loss.
        # Too few values may not capture the pattern; too many can overfit, train
        # more slowly, and need more data. The best size depends on the complexity
        # of the sequence, the number of input features, and the amount of data.

        self.lstm = nn.LSTM(input_size=input_size, hidden_size=5, num_layers=1, batch_first=True)

        # The LSTM returns a 5-value representation, not this task's final answer.
        # fc1 is the task-specific output layer: its 5 inputs must match hidden_size,
        # and its 1 output is the predicted next y-value. No activation follows it,
        # because this regression target can be either negative or positive.
        #
        # Common layers/losses after the final LSTM representation:
        #   Regression:          Linear(5, number_of_values) + MSE loss
        #   Binary class:        Linear(5, 1) + BCEWithLogitsLoss
        #   Single-label class:  Linear(5, number_of_classes) + CrossEntropyLoss
        #   Multi-label classes: Linear(5, number_of_labels) + BCEWithLogitsLoss
        # These classification losses apply the needed sigmoid or softmax-style
        # calculation internally during training, so no output activation is added.
        self.fc1 = nn.Linear(in_features=5, out_features=output_size)

    def forward(self, x):
        x, status = self.lstm(x)
        # LSTM output: (batch, 10, 5), one 5-value hidden summary per time step.
        # Keep each sample's final summary: (batch, 5), then map it to a prediction.
        x = x[:, -1, :]
        x = self.fc1(x)
        return x

#%% instantiate model, optimizer, and loss
model = TrigonometryModel()
# Random dummy batch used only to check the model's input and output shapes.
# Unlike X_train (1080, 10), it already includes the feature dimension: (2, 10, 1).
# Two samples enter the model, so model(input) returns two predictions: shape (2, 1).
input = torch.rand((2, 10, 1))
model(input).shape

#%% Loss and Optimizer
loss_fun = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
NUM_EPOCHS = 200

#%% Train
for epoch in range(NUM_EPOCHS):
    for j, (X, y) in enumerate(train_loader):
        optimizer.zero_grad()

        # From the input batch X, predict the next y-value using the model.
        # X has shape (batch, 10), but the model expects (batch, 10, 1) with
        # a feature dimension, the feature here being just 1 as discussed above
        y_pred = model(X.view(-1, 10, 1))

        # PyTorch tensors have reshape just like NumPy arrays. Change y from
        # (batch,) to (batch, 1); -1 tells PyTorch to infer the batch size.
        loss = loss_fun(y_pred, y.reshape(-1, 1))
        loss.backward()
        optimizer.step()
    if epoch % 50 == 0:
        print(f"Epoch: {epoch}, Loss: {loss.data}")
  

# %% Create Predictions
# make the predictions for the test set
test_set = TrigonometricDataset(X_test, y_test)
X_test_torch, y_test_torch = next(iter(test_loader))
with torch.no_grad():
    y_pred = model(torch.reshape(X_test_torch, (-1, 10, 1))).detach().squeeze().numpy()

# convert to numpy for sns compatibility
y_act = y_test_torch.numpy()
x_act = range(y_act.shape[0])
sns.lineplot(x=x_act, y=y_act, label = 'Actual',color='black')
sns.lineplot(x=x_act, y=y_pred, label = 'Predicted',color='red')

# %% correlation plot
sns.scatterplot(x=y_act, y=y_pred, label = 'Predicted',color='red', alpha=0.5)
# %%
