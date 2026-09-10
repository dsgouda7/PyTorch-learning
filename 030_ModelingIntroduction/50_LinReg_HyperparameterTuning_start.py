#%% packages
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import numpy as np
import pandas as pd
from skorch import NeuralNetRegressor
from sklearn.model_selection import GridSearchCV
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader 
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

#%% Dataset and Dataloader
class LinearRegressionDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

train_loader = DataLoader(dataset = LinearRegressionDataset(X_np, y_np), batch_size=2)


#%%
class LinearRegressionTorch(nn.Module):
    def __init__(self, input_size=1, output_size=1):
        super(LinearRegressionTorch, self).__init__()
        self.linear = nn.Linear(input_size, output_size)
    
    def forward(self, x):
        return self.linear(x)

input_dim = 1
output_dim = 1
model = LinearRegressionTorch(input_size=input_dim, output_size=output_dim)
model.train()

# %% Mean Squared Error
loss_fun = nn.MSELoss()

#%% Optimizer
learning_rate = 0.02
# test different values of too large 0.1 and too small 0.001
# best 0.02
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

#%%  The hyperparameters we generally tune for a NN
# 1. Network topology  => (number of layers, number of neurons per layer, activation types, etc)
# 2. Network objects => Loss function, optimizer
# 3. Model Training => (number of epochs, batch size, learning rate, etc)

# %% Animate the directional effects of common hyperparameters
# These normalized scores explain typical trade-offs; they are not hardware
# benchmarks. Actual results depend on the model, data, device, and framework.
HYPERPARAMETER_EFFECTS = [
    {
        'category': 'Network topology',
        'name': 'Number of layers',
        'values': ['1', '2', '4', '8', '16'],
        'metrics': {
            'Representation capacity': [25, 45, 65, 82, 92],
            'GPU memory': [15, 25, 43, 70, 95],
            'Training time': [18, 28, 48, 75, 98],
            'Inference latency': [12, 20, 38, 68, 95],
        },
        'note': 'More depth can learn richer features, but cost and optimization difficulty rise.',
    },
    {
        'category': 'Network topology',
        'name': 'Neurons per layer',
        'values': ['16', '64', '256', '1,024', '4,096'],
        'metrics': {
            'GPU utilization': [20, 38, 62, 85, 96],
            'GPU memory': [8, 18, 40, 72, 100],
            'Training time': [12, 25, 48, 78, 100],
            'Inference latency': [8, 18, 40, 74, 100],
        },
        'note': 'Wider layers expose more parallel work while increasing memory and compute.',
    },
    {
        'category': 'Network topology',
        'name': 'Activation function',
        'values': ['ReLU', 'LeakyReLU', 'GELU', 'SiLU', 'Tanh'],
        'metrics': {
            'Forward throughput': [95, 90, 72, 68, 55],
            'Gradient flow': [65, 82, 88, 90, 58],
            'Training stability': [70, 82, 90, 92, 62],
            'Activation memory': [35, 38, 55, 58, 48],
        },
        'note': 'Activation choice is categorical, not a low-to-high scale; architecture matters.',
    },
    {
        'category': 'Network objects',
        'name': 'Loss function',
        'values': ['MSE', 'MAE', 'Huber', 'Log-cosh'],
        'metrics': {
            'Outlier robustness': [20, 95, 80, 70],
            'Gradient smoothness': [95, 30, 85, 90],
            'Convergence speed': [85, 45, 78, 72],
            'Large-error penalty': [100, 35, 75, 68],
        },
        'note': 'Choose a loss whose error geometry matches the task and noise distribution.',
    },
    {
        'category': 'Network objects',
        'name': 'Optimizer',
        'values': ['SGD', 'SGD + momentum', 'RMSprop', 'Adam', 'AdamW'],
        'metrics': {
            'Early convergence': [30, 58, 72, 90, 88],
            'Training stability': [55, 75, 78, 85, 87],
            'Optimizer memory': [15, 30, 55, 80, 80],
            'Tuning simplicity': [35, 48, 60, 85, 80],
        },
        'note': 'Adaptive optimizers often converge quickly but keep extra state per parameter.',
    },
    {
        'category': 'Model training',
        'name': 'Number of epochs',
        'values': ['10', '50', '100', '500', '1,000'],
        'metrics': {
            'Optimizer iterations': [5, 20, 35, 75, 100],
            'Training time': [5, 20, 35, 75, 100],
            'Underfit reduction': [20, 55, 75, 92, 96],
            'Overfit risk': [5, 12, 25, 65, 90],
        },
        'note': 'More epochs add optimizer steps; validation metrics should decide when to stop.',
    },
    {
        'category': 'Model training',
        'name': 'Batch size',
        'values': ['1', '8', '32', '128', '512'],
        'metrics': {
            'GPU utilization': [10, 30, 62, 88, 98],
            'GPU memory': [5, 12, 30, 68, 100],
            'Iterations per epoch': [100, 75, 48, 22, 8],
            'Gradient stability': [15, 38, 65, 86, 96],
        },
        'note': 'Larger batches improve throughput and gradient consistency but consume memory.',
    },
    {
        'category': 'Model training',
        'name': 'Learning rate',
        'values': ['1e-5', '1e-4', '1e-3', '1e-2', '1e-1'],
        'metrics': {
            'Convergence speed': [5, 18, 55, 92, 40],
            'Training stability': [100, 95, 88, 70, 15],
            'Overshoot risk': [2, 5, 15, 40, 100],
            'Useful progress / step': [4, 18, 58, 90, 20],
        },
        'note': 'Too low is slow; too high overshoots or diverges. The useful range is task-specific.',
    },
    {
        'category': 'Model training',
        'name': 'Weight decay',
        'values': ['0', '1e-6', '1e-4', '1e-2', '1e-1'],
        'metrics': {
            'Regularization strength': [0, 8, 42, 80, 100],
            'Overfit resistance': [10, 20, 58, 88, 95],
            'Underfit risk': [3, 5, 12, 45, 95],
            'Parameter shrinkage': [2, 4, 15, 52, 100],
        },
        'note': 'Moderate decay can generalize better; excessive decay suppresses useful weights.',
    },
]


def create_hyperparameter_animation(profile, output_path, frame_count=30, fps=12):
    """Animate one hyperparameter's conceptual trade-offs and save them as a GIF."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    figure = plt.figure(figsize=(12, 7), facecolor='#f7f5ef')
    metric_axis = figure.add_axes([0.20, 0.24, 0.72, 0.48])
    value_axis = figure.add_axes([0.20, 0.12, 0.72, 0.055])
    colors = ['#246a73', '#e07a5f', '#3d5a80', '#81b29a']
    bars = metric_axis.barh(range(4), [0] * 4, color=colors, height=0.58)
    score_labels = [metric_axis.text(2, index, '0', va='center') for index in range(4)]
    value_marker, = value_axis.plot([0], [0], 'o', color='#d1495b', markersize=12)
    value_line, = value_axis.plot([], [], color='#3d405b', linewidth=3)

    title = figure.text(0.06, 0.93, '', fontsize=23, fontweight='bold', color='#1d3557')
    category = figure.text(0.06, 0.875, '', fontsize=12, color='#606c38')
    current_value = figure.text(0.94, 0.88, '', fontsize=15, ha='right', color='#9c2f24')
    note = figure.text(0.06, 0.035, '', fontsize=11, color='#343a40')
    figure.text(
        0.94,
        0.035,
        'Directional teaching scores, not benchmarks',
        fontsize=9,
        ha='right',
        color='#6c757d',
    )

    metric_names = list(profile['metrics'])
    values = profile['values']
    positions = np.arange(len(values))

    metric_axis.set_yticks(range(4), labels=metric_names)
    metric_axis.invert_yaxis()
    metric_axis.set_xlim(0, 100)
    metric_axis.set_xlabel('Normalized effect score (higher = more)')
    metric_axis.grid(axis='x', alpha=0.18)
    metric_axis.spines[['top', 'right', 'left']].set_visible(False)

    value_axis.set_xlim(-0.2, len(values) - 0.8)
    value_axis.set_ylim(-1, 1)
    value_axis.set_xticks(positions, labels=values)
    value_axis.set_yticks([])
    value_axis.spines[:].set_visible(False)
    value_line.set_data(positions, np.zeros(len(values)))

    title.set_text(profile['name'])
    category.set_text(profile['category'])
    note.set_text(profile['note'])

    def update(frame):
        position = frame / (frame_count - 1) * (len(values) - 1)
        lower_index = int(np.floor(position))
        upper_index = min(lower_index + 1, len(values) - 1)
        blend = position - lower_index

        scores = [
            metric_values[lower_index] * (1 - blend) + metric_values[upper_index] * blend
            for metric_values in profile['metrics'].values()
        ]
        selected_value = values[int(round(position))]

        for bar, label, score in zip(bars, score_labels, scores):
            bar.set_width(score)
            label.set_x(min(score + 2, 97))
            label.set_text(f'{score:.0f}')

        value_marker.set_data([position], [0])
        current_value.set_text(f'Value: {selected_value}')

        return (*bars, *score_labels, value_marker, value_line, title, category, current_value, note)

    animation = FuncAnimation(
        figure,
        update,
        frames=frame_count,
        interval=1000 / fps,
        repeat=True,
        blit=False,
    )
    animation.save(output_path, writer=PillowWriter(fps=fps), dpi=90)
    plt.close(figure)
    return animation


def generate_hyperparameter_animations(output_directory):
    """Generate one animation per hyperparameter and return the saved paths."""
    output_directory = Path(output_directory)
    animation_paths = []

    for profile in HYPERPARAMETER_EFFECTS:
        file_stem = profile['name'].lower().replace(' ', '-').replace('+', 'plus')
        output_path = output_directory / f'hyperparameter-{file_stem}.gif'
        create_hyperparameter_animation(profile, output_path)
        animation_paths.append(output_path)

    return animation_paths


if __name__ == '__main__':
    lesson_directory = Path(__file__).resolve().parent
    animation_paths = generate_hyperparameter_animations(lesson_directory / 'assets')
    print(f'Generated {len(animation_paths)} hyperparameter animations:')
    for animation_path in animation_paths:
        print(f'  {animation_path}')


#%% Hyperparameter Tuning with GridSearchCV

# skorch is a scikit-learn compatible neural network library that wraps PyTorch models
# we will use it for GridSearch of hyperparameters
net = NeuralNetRegressor(
    module=LinearRegressionTorch,
    max_epochs=100,
    lr=learning_rate,
    iterator_train__shuffle=True
)

net.set_params(train_split=False, verbose=0)

params = {
    # 0.1 is intentionally shown as unstable in the animation; on this
    # unscaled feature it diverges and produces non-finite cross-validation scores.
    'lr': [0.001, 0.01, 0.02],
    'max_epochs': [50, 100, 200]
}

gs = GridSearchCV(net, params, scoring='r2', cv=5)
gs.fit(X, y_true)
print(f"best score: {gs.best_score_}")
print(f"best parameters: {gs.best_params_}")