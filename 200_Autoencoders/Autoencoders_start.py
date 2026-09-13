#%% packages
from typing import OrderedDict
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
import numpy as np
import matplotlib.pyplot as plt
import torchvision.utils 

#%% Dataset and data loader
path_images = 'data/train'

# ========== What are Autoencoders and VAEs?  ============
# An autoencoder learns to represent an image using a smaller set of values.
# The encoder creates this compressed representation, called the latent representation.
# The decoder uses it to reconstruct an approximation of the original image.
#
# A VAE uses the same basic encoder -> latent space -> decoder structure.
# The main difference is that this autoencoder encodes each image as one fixed
# latent representation, while a VAE's encoder describes a probability
# distribution by producing a mean and a variance.
#
# A VAE samples a latent representation from that distribution and gives it to
# the decoder. It still learns by comparing the reconstruction with the input,
# but it also uses an extra loss that organizes the latent space. This makes it
# possible to sample new latent values and decode them into new images that
# resemble the training data.
#
# This script builds a standard image autoencoder, not a VAE.


# ============= Autoencoder for Images ===================
# as explained above, autoencoders can learn compact representations of images.
# The script below helps understand the structure and training of an autoencoder for images.
# Each source image starts as height x width x color channels. (standard image format)
# Generally speaking there are 3 color channels: Red, Green, and Blue (RGB).
# These transforms run in order before an image is returned by the dataset:
#   1. Resize makes every image 128 x 128 pixels so batches have a consistent size.
#      It does not crop away the outer parts of the image. Instead, it rescales
#      the entire image. Bilinear interpolation calculates each new pixel as a
#      weighted average of nearby pixels in the original image.
#      A large image keeps its overall content, but some fine detail is lost because
#      many original pixels must be summarized into fewer pixels. Forcing both
#      dimensions to 128 can also stretch an image whose original shape is not square.
#   2. ToTensor changes the RGB image into floating-point values between 0 and 1
#      and uses PyTorch's layout: (channels, height, width) = (3, 128, 128).
#   3. Normalize changes each color channel from [0, 1] to approximately [-1, 1].
#
# The image remains a 2D pixel grid here; it is not flattened into a vector yet.
# DataLoader groups 4 images by adding a batch dimension, producing
# (batch, channels, height, width) = (4, 3, 128, 128).
#
# From this point onward, "input image" means this processed tensor, not the raw
# file. The resize may already have removed fine detail. The autoencoder learns
# to reconstruct the processed 128 x 128 RGB image and cannot recover information
# that was lost before the neural network received it.
transform = transforms.Compose([
    transforms.Resize(
        (128, 128),
        # interpolation method for resizing the image
        # converts a large image into a smaller one
        # using bilinear interpolation (weighted average of nearby pixels)
        interpolation=transforms.InterpolationMode.BILINEAR,
        antialias=True,
    ),
    # Convert the image to a tensor and normalize it to [-1, 1] range
    transforms.ToTensor(),
    # Normalize the image to have mean 0.5 and standard deviation 0.5 for each channel
    # this is necessary because the network expects inputs roughly in the range [-1, 1]
    # NNs tend to perform better when inputs are centered around 0 and have similar scales
    transforms.Normalize(
        mean=(0.5, 0.5, 0.5),
        std=(0.5, 0.5, 0.5),
    ),
])

# In the LSTM example, TrigonometricDataset was a custom Dataset: we supplied
# the tensors and wrote __len__ and __getitem__ to define how one sample is read.
# ImageFolder is a ready-made Dataset that provides those methods for image files.
# It expects folders such as data/train/apples and data/train/bananas, scans the
# files, assigns a numeric label from the folder name, and applies `transform`
# whenever an image is requested. Each item is therefore (image_tensor, label).
#
# The label is useful for classification, but this autoencoder ignores it because
# the target is the input image itself. The neural network tries to reconstruct
# each processed image rather than predict whether it is an apple or a banana.
dataset = ImageFolder(root=path_images, transform=transform)

# DataLoader has the same role as in the LSTM example. It asks the Dataset for
# individual samples, optionally shuffles their order, and stacks them into
# batches. Here one batch contains:
#   images: (4, 3, 128, 128)
#   labels: (4,)
# DataLoader does not read image formats or resize pixels itself; ImageFolder and
# the transform pipeline handle that work before DataLoader combines the samples.
dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

# %% model class
LATENT_DIMS = 128

# the encoder class for the autoencoder
class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        # input shape is (3, 128, 128) for RGB images
        # CONV2D layer with 3 input channel, 6 output channels, kernel size 3, stride 1
        # 3 input channels for RGB images
        # 6 output channels for the convolutional layer (higher number of output channels
        # allows the network to capture more complex features) without increasing the
        # spatial dimensions (height , width) of the intermediate feature maps too much
        # kernel size indicates the sliding window size that will be processing the input image
        # stride indicates how many pixels the sliding window moves at each step

        # NOTE: an in depth understanding of convolutional layers is strictly not required to learn
        # generative models like transformers going forward but the basics of convolutional layers are still useful to understand.
        # especially in the context of how tensors are manipulated in convolutional neural networks.
        self.conv1 = nn.Conv2d(3, 6, 3) # output shape is (6, 126, 126) because kernel size is 3 and stride is 1
        self.conv2 = nn.Conv2d(6, 16, 3) # output shape is (16, 124, 124) because kernel size is 3 and stride is 1
        self.relu = nn.ReLU()
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(16 * 124 * 124, LATENT_DIMS) # 16*124*124 is the flattened size of the feature map after the second convolutional layer


    # standard forward pass through the encoder using all the layers defined in the ctor
    # ReLU after each convolution adds nonlinearity so stacked layers can learn complex visual patterns instead of one linear mapping.
    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.flatten(x)
        x = self.fc(x)
        return x

# the decoder class for the autoencoder
class Decoder(nn.Module):
    # `-> None` is an optional type hint indicating that __init__ returns no value.
    def __init__(self) -> None:
        super().__init__()
        self.fc = nn.Linear(LATENT_DIMS, 16 * 124 * 124) # reverse the compression done in the encoder
        self.conv2 = nn.ConvTranspose2d(16, 6, 3) # output shape is (6, 126, 126) because kernel size is 3 and stride is 1
        self.conv1 = nn.ConvTranspose2d(6, 3, 3) # output shape is (3, 128, 128) because kernel size is 3 and stride is 1
        self.relu = nn.ReLU()
        self.flatten = nn.Flatten()

    # ReLU after each convolution adds nonlinearity so stacked layers can learn complex visual patterns instead of one linear mapping.
    def forward(self, x):
        x = self.fc(x)
        x = x.reshape(-1, 16, 124, 124)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.conv1(x)
        x = self.relu(x)
        return x


# Now build the Autoencoder class using the Encoder and Decoder defined above
class Autoencoder(nn.Module):
    def __init__(self)->None:
        super().__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()

    def forward(self, x):
        return self.decoder(self.encoder(x))


#%% init model, loss function, optimizer
model = Autoencoder()

# standard optimizer initialization that we do for a typical ML/NN model
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

NUM_EPOCHS = 30

for epoch in range(NUM_EPOCHS):
    losses_epoch = []
    for batch_idx, (data, _) in enumerate(dataloader):
        output = model(data)

        # Both tensors are processed 128 x 128 RGB images. The model is trained
        # to reproduce `data`, not the untouched image file on disk.
        loss = F.mse_loss(output, data)
        losses_epoch.append(loss.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    print(f"Epoch: {epoch} \tLoss: {np.mean(losses_epoch)}")  

# %% visualise original and reconstructed images
def show_image(img):
    img = 0.5 * (img + 1)  # change [-1, 1] back to [0, 1]
    img = img.clamp(0, 1)
    npimg = img.detach().cpu().numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))

images, labels = next(iter(dataloader))
print('original')
plt.rcParams["figure.figsize"] = (20,3)
show_image(torchvision.utils.make_grid(images))

# %% latent space
print('latent space')
latent_img = model.encoder(images)
latent_img = latent_img.view(-1, 1, 8, 16)
show_image(torchvision.utils.make_grid(latent_img))
#%%
print('reconstructed')
show_image(torchvision.utils.make_grid(model(images)))


# %% Compression rate
image_size = images.shape[1] * images.shape[2] * images.shape[3]
compression_rate = (1 - LATENT_DIMS / image_size) * 100
compression_rate
