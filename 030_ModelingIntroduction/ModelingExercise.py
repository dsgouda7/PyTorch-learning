#!/usr/bin/env python
# coding: utf-8

# # Linear Regression in PyTorch: From Tensors to DataLoaders
# 
# This notebook combines the `00`, `10`, `20`, `30`, `40`, and `50` modeling exercises into one progression.
# 
# We will implement and manage the same one-feature regression problem in six stages:
# 
# 1. tensors and autograd from scratch;
# 2. an `nn.Module` trained on the full dataset;
# 3. manually shuffled mini-batches;
# 4. a reusable `Dataset` and `DataLoader`;
# 5. saving and loading learned model state;
# 6. identifying the main categories of hyperparameters.
# 
# Each training section starts from fresh parameters, so its result does not depend on state left behind by an earlier example.

# ## 1. Source files and execution order
# 
# The original exercises are useful as references. This check documents exactly which files were consolidated and prevents a missing or accidentally renamed source file from going unnoticed.

# In[1]:


from pathlib import Path

required_prefixes = ("00", "10", "20", "30", "40", "50")
lesson_directory = Path.cwd() / "030_ModelingIntroduction"
if not lesson_directory.exists():
    lesson_directory = Path.cwd()

selected_source_files = sorted(
    path
    for path in lesson_directory.glob("*_start.py")
    if path.name.startswith(required_prefixes)
)

assert [path.name[:2] for path in selected_source_files] == list(required_prefixes), (
    "Expected the 00, 10, 20, 30, 40, and 50 start files."
)

source_texts = {path.name: path.read_text(encoding="utf-8") for path in selected_source_files}
print("Combined in this order:")
for path in selected_source_files:
    print(f"  {path.name}")


# ## 2. Shared setup and data preparation
# 
# All four approaches solve the same problem: predict fuel economy (`mpg`) from vehicle weight (`wt`). The CSV columns initially become one-dimensional NumPy arrays with shape `(n,)`. A linear layer expects rows of feature vectors, so we reshape each array to `(n, 1)` before creating `float32` tensors.
# 
# Using one shared representation makes the learned slopes and intercepts directly comparable across sections.

# In[2]:


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
from sklearn.linear_model import LinearRegression
from torch.utils.data import DataLoader, Dataset

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
sns.set_theme(style="whitegrid")

cars_file = "https://gist.githubusercontent.com/noamross/e5d3e859aa0c794be10b/raw/b999fb4425b54c63cab088c0ce2c0d6ce961a563/cars.csv"
cars = pd.read_csv(cars_file)

feature_values = cars["wt"].to_numpy(dtype=np.float32).reshape(-1, 1)
target_values = cars["mpg"].to_numpy(dtype=np.float32).reshape(-1, 1)
features = torch.from_numpy(feature_values)
targets = torch.from_numpy(target_values)

assert features.shape == targets.shape == (len(cars), 1)
assert features.dtype == targets.dtype == torch.float32
cars.head()


# In[3]:


fig, ax = plt.subplots(figsize=(8, 5))
sns.scatterplot(data=cars, x="wt", y="mpg", ax=ax, label="Cars")
sns.regplot(data=cars, x="wt", y="mpg", scatter=False, ax=ax, color="tab:red", label="Statistical fit")
ax.set(title="Fuel economy decreases as vehicle weight increases", xlabel="Weight (1,000 lb)", ylabel="Miles per gallon")
plt.show()


# ## 3. Linear regression from scratch
# 
# A one-feature linear model is
# 
# $$\hat{y} = xw + b$$
# 
# where `w` is the slope and `b` is the intercept. PyTorch's autograd engine computes the derivatives of mean squared error with respect to both parameters.
# 
# The training cycle is always the same:
# 
# 1. calculate predictions and loss;
# 2. call `loss.backward()` to populate gradients;
# 3. update parameters without tracking that update in the computation graph;
# 4. clear gradients before the next iteration, because PyTorch accumulates them by default.

# In[4]:


torch.manual_seed(SEED)
manual_weight = torch.rand((1, 1), requires_grad=True)
manual_bias = torch.rand(1, requires_grad=True)
manual_losses = []

manual_epochs = 3_000
manual_learning_rate = 0.01

for epoch in range(manual_epochs):
    manual_predictions = features @ manual_weight + manual_bias
    manual_loss = torch.mean((manual_predictions - targets) ** 2)
    manual_loss.backward()

    with torch.no_grad():
        manual_weight -= manual_learning_rate * manual_weight.grad
        manual_bias -= manual_learning_rate * manual_bias.grad
        manual_weight.grad.zero_()
        manual_bias.grad.zero_()

    manual_losses.append(manual_loss.item())

print(f"Final loss: {manual_losses[-1]:.4f}")
print(f"Weight: {manual_weight.item():.4f}; bias: {manual_bias.item():.4f}")


# In[5]:


with torch.inference_mode():
    manual_fitted_values = (features @ manual_weight + manual_bias).squeeze().numpy()

sklearn_model = LinearRegression().fit(feature_values, target_values.ravel())

fig, axes = plt.subplots(1, 2, figsize=(13, 4))
axes[0].plot(manual_losses)
axes[0].set(title="From-scratch training loss", xlabel="Epoch", ylabel="MSE")
sns.scatterplot(data=cars, x="wt", y="mpg", ax=axes[1])
axes[1].plot(cars["wt"], manual_fitted_values, color="tab:red")
axes[1].set(title="From-scratch fitted line")
plt.tight_layout()
plt.show()

print(f"scikit-learn slope: {sklearn_model.coef_[0]:.4f}")
print(f"scikit-learn intercept: {sklearn_model.intercept_:.4f}")


# ## 4. Encapsulating the model with `nn.Module`
# 
# Every PyTorch model and layer derives from `nn.Module`. The constructor declares the layers and `forward` describes how input flows through them. Here, `nn.Linear(1, 1)` registers the same learnable weight and bias used in the manual example.
# 
# `nn.MSELoss` calculates a scalar error, while `torch.optim.SGD` receives references to the model's parameters and later updates them. The data is still processed all at once, so this remains **full-batch gradient descent**.
# 
# ### Training components at a glance
# 
# ![Overview of the model, loss, gradient computation, and optimizer in a PyTorch training loop](assets/pytorch-model-training-components.png)
# 
# The supplied infographic is a useful overview of the repeated training loop. Read two labels with care:
# 
# - “The loss function creates gradients” is shorthand. The loss function only calculates a loss tensor; `loss.backward()` asks autograd to traverse that tensor's computation graph and write each parameter's `.grad`.
# - `model.eval()` changes the behavior of layers such as dropout and batch normalization. It does **not** lock parameters or disable gradient tracking.
# 
# ### Copilot image-generation prompt
# 
# Use the following prompt when generating a more precise replacement for the overview image:
# 
# ```text
# Create a clean 16:9 educational infographic titled “How PyTorch Training Shares Parameter State.” Use a white background, crisp dark text, and restrained blue, orange, purple, and green accents. Avoid decorative characters and avoid tiny text.
# 
# Show four distinct components: Model (`nn.Module`), Loss (`nn.MSELoss`), Autograd, and Optimizer (`torch.optim.SGD`). Put one central object labeled “Shared Parameter objects: weight, bias” between Model, Autograd, and Optimizer.
# 
# Draw these exact relationships:
# 1. Model registers the Parameter objects.
# 2. `optimizer = SGD(model.parameters(), lr=...)` makes the optimizer retain references to those exact same Parameter objects; it does not copy them.
# 3. `predictions = model(inputs)` dynamically builds an autograd computation graph from the parameters to predictions.
# 4. `loss = loss_fn(predictions, targets)` extends the graph to a scalar loss. Clearly label: “Loss function stores no reference to the model or optimizer.”
# 5. `loss.backward()` makes Autograd write gradients into each shared Parameter object's `.grad` field. Clearly label: “backward computes gradients; it does not update parameter values.”
# 6. `optimizer.step()` reads `.grad` and mutates the shared parameter values.
# 7. `optimizer.zero_grad()` clears or resets `.grad` before the next iteration.
# 
# Include a warning panel titled “PyTorch design trade-off: implicit ownership.” State that the model registers parameters but is not their exclusive mutable owner: Autograd writes `.grad`, the optimizer clears `.grad` and changes values, and user code can also mutate values under `torch.no_grad()`. List three risks: stale accumulated gradients, optimizing the wrong parameter collection, and multiple optimizers silently sharing parameters.
# 
# Do not claim that the loss function creates gradients, that `backward()` updates weights, that `eval()` freezes parameters, or that training always improves the model. Make arrows directional and keep every code label exactly spelled.
# ```

# In[6]:


class LinearRegressionTorch(nn.Module):
    def __init__(self, input_size=1, output_size=1):
        super().__init__()
        self.linear = nn.Linear(input_size, output_size)

    def forward(self, inputs):
        return self.linear(inputs)


def model_parameters(model):
    return model.linear.weight.item(), model.linear.bias.item()


torch.manual_seed(SEED)
full_batch_model = LinearRegressionTorch()
loss_function = nn.MSELoss()
full_batch_optimizer = torch.optim.SGD(full_batch_model.parameters(), lr=0.01)

registered_parameters = list(full_batch_model.parameters())
optimizer_parameters = [
    parameter
    for parameter_group in full_batch_optimizer.param_groups
    for parameter in parameter_group["params"]
]
optimizer_uses_same_parameters = (
    len(registered_parameters) == len(optimizer_parameters)
    and all(
        model_parameter is optimizer_parameter
        for model_parameter, optimizer_parameter in zip(
            registered_parameters, optimizer_parameters
        )
    )
)

assert optimizer_uses_same_parameters
assert len(list(loss_function.parameters())) == 0
print(f"Optimizer references model parameters: {optimizer_uses_same_parameters}")
print(f"Loss function trainable parameters: {len(list(loss_function.parameters()))}")

full_batch_losses = []
full_batch_weights = []
full_batch_biases = []

for epoch in range(3_000):
    # The optimizer clears .grad on the model parameters it references.
    full_batch_optimizer.zero_grad()

    # These operations dynamically connect parameters, predictions, and loss.
    predictions = full_batch_model(features)
    loss = loss_function(predictions, targets)

    if epoch == 0:
        print(f"Prediction tracks gradients: {predictions.requires_grad}")
        print(f"Loss autograd node: {type(loss.grad_fn).__name__}")

    # Autograd writes gradients to Parameter.grad; it does not update values.
    loss.backward()

    if epoch == 0:
        parameter_values_before_step = {
            name: parameter.detach().clone()
            for name, parameter in full_batch_model.named_parameters()
        }
        first_gradients = {
            name: parameter.grad.detach().clone()
            for name, parameter in full_batch_model.named_parameters()
        }
        print(f"Model gradients after backward: {first_gradients}")

    # The optimizer reads Parameter.grad and mutates those Parameter values.
    full_batch_optimizer.step()

    if epoch == 0:
        changed_parameters = {
            name: not torch.equal(parameter_values_before_step[name], parameter)
            for name, parameter in full_batch_model.named_parameters()
        }
        print(f"Parameters changed after optimizer.step(): {changed_parameters}")

    weight, bias = model_parameters(full_batch_model)
    full_batch_losses.append(loss.item())
    full_batch_weights.append(weight)
    full_batch_biases.append(bias)

print(f"Final loss: {full_batch_losses[-1]:.4f}")
print(f"Weight: {full_batch_weights[-1]:.4f}; bias: {full_batch_biases[-1]:.4f}")


# In[7]:


with torch.inference_mode():
    full_batch_fitted_values = full_batch_model(features).squeeze().numpy()

fig, axes = plt.subplots(1, 3, figsize=(16, 4))
axes[0].plot(full_batch_losses)
axes[0].set(title="Full-batch loss", xlabel="Epoch", ylabel="MSE")
axes[1].plot(full_batch_weights, label="Weight")
axes[1].plot(full_batch_biases, label="Bias")
axes[1].set(title="Parameter updates", xlabel="Epoch")
axes[1].legend()
sns.scatterplot(data=cars, x="wt", y="mpg", ax=axes[2])
axes[2].plot(cars["wt"], full_batch_fitted_values, color="tab:red")
axes[2].set(title="Full-batch fitted line")
plt.tight_layout()
plt.show()


# ## 5. Shuffled mini-batches by hand
# 
# Full-batch training computes one gradient from every sample. Mini-batch training instead updates the model several times per epoch using smaller subsets.
# 
# `torch.randperm` creates shuffled row indices. For each mini-batch, gradients must be cleared **before** the forward and backward passes. We store one mean loss and one parameter snapshot per epoch so every history has the same length and can be plotted safely.

# In[8]:


torch.manual_seed(SEED)
manual_batch_model = LinearRegressionTorch()
manual_batch_optimizer = torch.optim.SGD(manual_batch_model.parameters(), lr=0.01)
manual_batch_losses = []
manual_batch_weights = []
manual_batch_biases = []

batch_size = 8
mini_batch_epochs = 500

for epoch in range(mini_batch_epochs):
    permutation = torch.randperm(len(features))
    batch_losses = []

    for start in range(0, len(features), batch_size):
        indices = permutation[start:start + batch_size]
        batch_features = features[indices]
        batch_targets = targets[indices]

        manual_batch_optimizer.zero_grad()
        predictions = manual_batch_model(batch_features)
        loss = loss_function(predictions, batch_targets)
        loss.backward()
        manual_batch_optimizer.step()
        batch_losses.append(loss.item())

    weight, bias = model_parameters(manual_batch_model)
    manual_batch_losses.append(float(np.mean(batch_losses)))
    manual_batch_weights.append(weight)
    manual_batch_biases.append(bias)

assert len(manual_batch_losses) == len(manual_batch_weights) == mini_batch_epochs
print(f"Final mean batch loss: {manual_batch_losses[-1]:.4f}")


# In[9]:


with torch.inference_mode():
    manual_batch_fitted_values = manual_batch_model(features).squeeze().numpy()

fig, axes = plt.subplots(1, 3, figsize=(16, 4))
axes[0].plot(manual_batch_losses)
axes[0].set(title="Manual mini-batch loss", xlabel="Epoch", ylabel="Mean batch MSE")
axes[1].plot(manual_batch_weights, label="Weight")
axes[1].plot(manual_batch_biases, label="Bias")
axes[1].set(title="Parameter updates", xlabel="Epoch")
axes[1].legend()
sns.scatterplot(data=cars, x="wt", y="mpg", ax=axes[2])
axes[2].plot(cars["wt"], manual_batch_fitted_values, color="tab:red")
axes[2].set(title="Manual mini-batch fitted line")
plt.tight_layout()
plt.show()


# ## 6. Replacing batching code with `Dataset` and `DataLoader`
# 
# A `Dataset` defines how many examples exist and how to retrieve one example. A `DataLoader` turns that dataset into an iterable of batches and can handle shuffling for us. It should be constructed before training and consumed directly inside the training loop, replacing manual slicing or permutation code.
# 
# A custom map-style dataset implements:
# 
# - `__len__`: the number of examples;
# - `__getitem__`: the feature-target pair at a given index.
# 
# A dataset may return NumPy arrays; PyTorch's default collation converts them to tensors while forming a batch. This notebook stores tensors directly so the conversion and `float32` dtype are explicit at the dataset boundary.
# 
# The optimization steps remain unchanged. Only the input pipeline becomes reusable and easier to scale.

# In[10]:


class CarsDataset(Dataset):
    def __init__(self, feature_data, target_data):
        self.feature_data = feature_data
        self.target_data = target_data

    def __len__(self):
        return len(self.feature_data)

    def __getitem__(self, index):
        return self.feature_data[index], self.target_data[index]


cars_dataset = CarsDataset(features, targets)
batch_generator = torch.Generator().manual_seed(SEED)
cars_loader = DataLoader(
    cars_dataset,
    batch_size=batch_size,
    shuffle=True,
    generator=batch_generator,
)

sample_features, sample_targets = next(iter(cars_loader))
assert len(cars_dataset) == len(cars)
assert sample_features.shape[1:] == sample_targets.shape[1:] == (1,)
print(f"Dataset size: {len(cars_dataset)}")
print(f"Example batch shapes: {sample_features.shape}, {sample_targets.shape}")


# In[11]:


torch.manual_seed(SEED)
dataloader_model = LinearRegressionTorch()
dataloader_optimizer = torch.optim.SGD(dataloader_model.parameters(), lr=0.01)
dataloader_losses = []

for epoch in range(500):
    batch_losses = []

    for batch_features, batch_targets in cars_loader:
        dataloader_optimizer.zero_grad()
        predictions = dataloader_model(batch_features)
        loss = loss_function(predictions, batch_targets)
        loss.backward()
        dataloader_optimizer.step()
        batch_losses.append(loss.item())

    dataloader_losses.append(float(np.mean(batch_losses)))

with torch.inference_mode():
    dataloader_fitted_values = dataloader_model(features).squeeze().numpy()

assert dataloader_fitted_values.shape == (len(cars),)
assert np.isfinite(dataloader_fitted_values).all()
print(f"Final mean batch loss: {dataloader_losses[-1]:.4f}")


# In[12]:


fig, axes = plt.subplots(1, 2, figsize=(13, 4))
axes[0].plot(dataloader_losses)
axes[0].set(title="DataLoader training loss", xlabel="Epoch", ylabel="Mean batch MSE")
sns.scatterplot(data=cars, x="wt", y="mpg", ax=axes[1])
axes[1].plot(cars["wt"], dataloader_fitted_values, color="tab:red")
axes[1].set(title="DataLoader fitted line")
plt.tight_layout()
plt.show()


# ## 7. Saving and loading learned state
# 
# `model.state_dict()` is an ordered mapping of parameter and persistent-buffer names to tensors. Saving this mapping is generally more portable than serializing the entire Python model object.
# 
# Loading requires the same model architecture:
# 
# 1. construct a fresh model instance;
# 2. load the saved state dictionary;
# 3. call `eval()` before inference;
# 4. use `torch.inference_mode()` while calculating predictions.
# 
# `eval()` is often described as “locking” a model, but that is inaccurate. It only switches layers such as dropout and batch normalization to evaluation behavior. It does not freeze parameters and does not disable autograd.
# 
# The example uses a temporary directory so running the notebook does not overwrite a repository checkpoint.

# In[13]:


from tempfile import TemporaryDirectory

with TemporaryDirectory() as temporary_directory:
    checkpoint_path = Path(temporary_directory) / "linear_regression_model.pth"
    torch.save(dataloader_model.state_dict(), checkpoint_path)

    reloaded_model = LinearRegressionTorch()
    saved_state = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=True,
    )
    reloaded_model.load_state_dict(saved_state)
    reloaded_model.eval()

    dataloader_model.eval()
    with torch.inference_mode():
        original_predictions = dataloader_model(features)
        reloaded_predictions = reloaded_model(features)

assert torch.equal(original_predictions, reloaded_predictions)
print(f"Saved state keys: {list(saved_state)}")
print("Reloaded predictions exactly match the trained model.")


# ## 8. Choosing hyperparameters
# 
# Model **parameters** such as weights and biases are learned from data. **Hyperparameters** are choices made before or around training that control the model and learning process.
# 
# The main categories are:
# 
# | Category | Examples |
# |---|---|
# | Model architecture | Number of layers, units per layer, activation functions |
# | Objective and update rule | Loss-function choice, optimizer family, momentum, weight decay |
# | Training process | Number of epochs, batch size, learning rate |
# 
# The loss and optimizer objects are not themselves parameter state owned by the model. Their selected algorithms and constructor settings are training choices that can be tuned. Keep these choices explicit so experiments are reproducible.
# 

# ### Visualizing one choice at a time
# 
# Each animation isolates one hyperparameter while holding the interpretation of the other choices fixed. Follow the red marker across the candidate values and compare the directional effect scores. These normalized scores are teaching aids, not measured benchmarks; actual behavior depends on the data, model, hardware, and software stack.
# 
# #### Model architecture
# 
# Depth and width are ordered sweeps: moving right means adding layers or units. Activation functions are categorical, so read that panel as a comparison among alternatives rather than a low-to-high progression.
# 
# | Number of layers | Neurons per layer | Activation function |
# |---|---|---|
# | ![Effects of changing the number of layers](assets/hyperparameter-number-of-layers.gif) | ![Effects of changing neurons per layer](assets/hyperparameter-neurons-per-layer.gif) | ![Effects of changing the activation function](assets/hyperparameter-activation-function.gif) |
# 
# #### Objective and update rule
# 
# Loss functions encode what counts as an expensive error. Optimizers determine how gradients become parameter updates. Both panels compare categorical alternatives; optimizer memory refers to state such as momentum or adaptive moments.
# 
# | Loss function | Optimizer |
# |---|---|
# | ![Effects of choosing a loss function](assets/hyperparameter-loss-function.gif) | ![Effects of choosing an optimizer](assets/hyperparameter-optimizer.gif) |
# 
# #### Training process
# 
# Epochs, batch size, learning rate, and weight decay interact. More is not automatically better: validation performance should select the useful operating range, and the learning-rate panel explicitly shows the instability caused by oversized steps.
# 
# | Number of epochs | Batch size |
# |---|---|
# | ![Effects of changing the number of epochs](assets/hyperparameter-number-of-epochs.gif) | ![Effects of changing batch size](assets/hyperparameter-batch-size.gif) |
# 
# | Learning rate | Weight decay |
# |---|---|
# | ![Effects of changing the learning rate](assets/hyperparameter-learning-rate.gif) | ![Effects of changing weight decay](assets/hyperparameter-weight-decay.gif) |
# 
# Treat these plots as hypotheses to test. In a real experiment, change a controlled set of choices, measure training and validation metrics, and keep the data split and random seeds consistent.

# ## 9. Compare the implementations
# 
# All approaches optimize the same objective and should learn similar parameters. Small differences are expected because mini-batch methods update parameters using shuffled subsets rather than one exact full-dataset gradient.
# 
# The scikit-learn solution provides a closed-form baseline for checking the learned values.

# In[14]:


manual_batch_weight, manual_batch_bias = model_parameters(manual_batch_model)
dataloader_weight, dataloader_bias = model_parameters(dataloader_model)

results = pd.DataFrame(
    {
        "method": ["Manual tensors", "nn.Module full batch", "Manual mini-batches", "Dataset + DataLoader", "scikit-learn"],
        "slope": [manual_weight.item(), full_batch_weights[-1], manual_batch_weight, dataloader_weight, sklearn_model.coef_[0]],
        "intercept": [manual_bias.item(), full_batch_biases[-1], manual_batch_bias, dataloader_bias, sklearn_model.intercept_],
    }
)

assert np.isfinite(results[["slope", "intercept"]].to_numpy()).all()
results.round(4)


# ## 10. Optional model graph
# 
# `torchview` can render an individual `nn.Module` computation graph. This complements the conceptual training-loop infographic: `torchview` focuses on tensor operations inside the model, while the infographic explains cooperation among the model, loss, autograd, and optimizer.
# 
# This visualization is optional because training and evaluation do not depend on graph-rendering tooling.

# In[15]:


try:
    from torchview import draw_graph

    model_graph = draw_graph(dataloader_model, input_data=features)
    model_graph.visual_graph
except ImportError:
    print("Optional dependency missing. Install torchview to display the model graph.")


# ## 11. Export the notebook as Python
# 
# `nbconvert` is listed in the project requirements so this notebook can be validated from a clean kernel and exported reproducibly. From the repository root, run:
# 
# ```powershell
# python -m jupyter nbconvert --to script 030_ModelingIntroduction/ModelingExercise.ipynb --output ModelingExercise --output-dir 030_ModelingIntroduction
# ```
# 
# The generated `ModelingExercise.py` preserves markdown explanations as commented blocks. Treat the notebook as the primary source and regenerate the script after notebook edits.

# ## Key takeaways
# 
# - Autograd removes the need to derive gradients by hand, but `backward()` computes gradients; it does not update parameter values.
# - `nn.Module` registers parameters and groups forward-pass logic into a reusable model.
# - PyTorch coordinates training through shared mutable `Parameter` objects: autograd writes `.grad`, optimizers clear gradients and update values, and user code can mutate them too. This flexibility makes ownership less explicit.
# - Full-batch training makes one update per epoch; mini-batch training makes several noisier updates that can scale to larger datasets.
# - Hyperparameters are chosen before training; learned weights and biases are model parameters, not hyperparameters.
# - `Dataset` describes individual examples, while `DataLoader` handles batching, collation, and shuffling.
# - Save a model's `state_dict`, recreate the architecture before loading, and use `eval()` together with `torch.inference_mode()` for inference.
# - `eval()` changes layer behavior; it does not freeze parameters or disable gradient tracking.
