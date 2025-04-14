
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch.optim as optim
import wandb
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os
import numpy as np
from torch.utils.data import DataLoader, random_split, Subset
from sklearn.model_selection import StratifiedShuffleSplit
import os
import zipfile
import urllib.request
import shutil



# Download and unzip the dataset
if not os.path.exists('nature_12K'):
    url = "https://storage.googleapis.com/wandb_datasets/nature_12K.zip"
    zip_path = "nature_12K.zip"
    print("Downloading dataset...")
    urllib.request.urlretrieve(url, zip_path)
    print("Unzipping dataset...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall('.')
    os.remove(zip_path)

# visualize_dataset.py
import os
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict


def load_dataset(data_dir):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    dataset = datasets.ImageFolder(root=data_dir, transform=transform)
    loader = DataLoader(dataset, batch_size=64, shuffle=True)
    return dataset, loader


def show_images_per_class(dataset, samples_per_class=10):
    class_map = defaultdict(list)

    for img, label in dataset:
        if len(class_map[label]) < samples_per_class:
            class_map[label].append(img)
        if all(len(v) >= samples_per_class for v in class_map.values()):
            break

    for class_idx, images in class_map.items():
        fig, axes = plt.subplots(1, samples_per_class, figsize=(20, 2))
        fig.suptitle(f"Class: {dataset.classes[class_idx]}", fontsize=14)
        for i, img in enumerate(images):
            axes[i].imshow(np.transpose(img.numpy(), (1, 2, 0)))
            axes[i].axis('off')
        plt.show()


def print_dataset_info(dataset):
    class_counts = defaultdict(int)
    for _, label in dataset:
        class_counts[label] += 1

    print("Total images:", len(dataset))
    print("Number of classes:", len(dataset.classes))
    for class_idx, class_name in enumerate(dataset.classes):
        print(f"{class_name}: {class_counts[class_idx]} images")

    sample_img, _ = dataset[0]
    print("Sample image size:", sample_img.shape)


# Run this only when data is available locally
if __name__ == "__main__":
    data_dir = "/content/inaturalist_12K/train"  
    dataset, loader = load_dataset(data_dir)
    print_dataset_info(dataset)
    show_images_per_class(dataset, samples_per_class=10)



class SimpleCNN(nn.Module):
    def __init__(self, input_channels=3, num_classes=10, 
                 filters_per_layer=None, filter_sizes=None, 
                 activation_fn=nn.ReLU(), dense_neurons=128,
                 input_size=(224, 224)):
        super(SimpleCNN, self).__init__()
        
        self.input_size = input_size
        
        # Default configurations if not provided
        if filters_per_layer is None:
            filters_per_layer = [32, 64, 128, 256, 512]
        
        if filter_sizes is None:
            filter_sizes = [3, 3, 3, 3, 3]
        
        assert len(filters_per_layer) == 5, "Must provide exactly 5 filter counts"
        assert len(filter_sizes) == 5, "Must provide exactly 5 filter sizes"
        
        # Store configurations for computation analysis
        self.filters_per_layer = filters_per_layer
        self.filter_sizes = filter_sizes
        self.dense_neurons = dense_neurons
        
        # First convolutional block
        self.conv1 = nn.Conv2d(input_channels, filters_per_layer[0], kernel_size=filter_sizes[0], padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Second convolutional block
        self.conv2 = nn.Conv2d(filters_per_layer[0], filters_per_layer[1], kernel_size=filter_sizes[1], padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Third convolutional block
        self.conv3 = nn.Conv2d(filters_per_layer[1], filters_per_layer[2], kernel_size=filter_sizes[2], padding=1)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Fourth convolutional block
        self.conv4 = nn.Conv2d(filters_per_layer[2], filters_per_layer[3], kernel_size=filter_sizes[3], padding=1)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Fifth convolutional block
        self.conv5 = nn.Conv2d(filters_per_layer[3], filters_per_layer[4], kernel_size=filter_sizes[4], padding=1)
        self.pool5 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Store the activation function
        self.activation = activation_fn
        
        # Calculate the flattened features size
        self._calculate_flatten_size()
        
        # Dense layers
        self.fc1 = nn.Linear(self.flattened_size, dense_neurons)
        self.fc2 = nn.Linear(dense_neurons, num_classes)
    
    def _calculate_flatten_size(self):
        # Create a dummy input to calculate the size after convolutions
        x = torch.zeros(1, 3, self.input_size[0], self.input_size[1])
        
        # Apply the convolutional layers
        x = self.activation(self.conv1(x))
        x = self.pool1(x)
        
        x = self.activation(self.conv2(x))
        x = self.pool2(x)
        
        x = self.activation(self.conv3(x))
        x = self.pool3(x)
        
        x = self.activation(self.conv4(x))
        x = self.pool4(x)
        
        x = self.activation(self.conv5(x))
        x = self.pool5(x)
        
        # Get the flattened size
        self.flattened_size = x.numel()
    
    def forward(self, x):
        # First convolutional block
        x = self.activation(self.conv1(x))
        x = self.pool1(x)
        
        # Second convolutional block
        x = self.activation(self.conv2(x))
        x = self.pool2(x)
        
        # Third convolutional block
        x = self.activation(self.conv3(x))
        x = self.pool3(x)
        
        # Fourth convolutional block
        x = self.activation(self.conv4(x))
        x = self.pool4(x)
        
        # Fifth convolutional block
        x = self.activation(self.conv5(x))
        x = self.pool5(x)
        
        # Flatten the output
        x = x.view(x.size(0), -1)
        
        # Dense layer
        x = self.activation(self.fc1(x))
        
        # Output layer
        x = self.fc2(x)
        
        return x

