import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, Subset, random_split
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torchvision.datasets import ImageFolder
from PIL import Image
import wandb
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from tqdm import tqdm

!pip install wandb

#check GPU working?
import torch
torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else "No GPU"

#My wandb login key
!wandb login 6001619563748a57b4114b0bb090fd4129ba6122

# Function to get activation function
def get_activation(name):
    if name == 'relu':
        return nn.ReLU()
    elif name == 'gelu':
        return nn.GELU()
    elif name == 'silu':
        return nn.SiLU()
    elif name == 'mish':
        return nn.Mish()
    else:
        return nn.ReLU()  # Default


# Data preparation functions 
# split train data into val =0.2 * train  
def get_dataset_and_loaders(data_dir, train_transform, val_transform, batch_size=64, val_split=0.2):
    full_train_dataset = ImageFolder(os.path.join(data_dir, 'train'), transform=train_transform)
    
    # Get class indices
    classes = full_train_dataset.classes
    class_to_idx = full_train_dataset.class_to_idx
    
    # Create indices for stratified split
    targets = np.array(full_train_dataset.targets)
    
    # Split indices for each class to maintain class balance
    train_indices, val_indices = [], []
    
    for class_idx in range(len(classes)):
        class_indices = np.where(targets == class_idx)[0]
        # data split
        n_val = int(len(class_indices) * val_split)     
        
        # Shuffle indices
        np.random.shuffle(class_indices)
        
        # Split into train and validation
        val_indices.extend(class_indices[:n_val])
        train_indices.extend(class_indices[n_val:])
    
    # Create train and validation subsets
    train_dataset = Subset(full_train_dataset, train_indices)
    val_dataset = Subset(full_train_dataset, val_indices)
    
    # For validation set, we want to use a different transform
    val_dataset.dataset.transform = val_transform
    

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    
    return train_loader, val_loader, classes


# # Configure the sweep
sweep_config = {
    'method': 'random', 

    'parameters': {
        'learning_rate': {
            'values': [0.001, 0.0005, 0.0001,0.01]
        },
        'train_batch_size': {
            'values': [64, 128] 
        },
        'epochs': {
            'values': [10,20]  
        },
        'activation': {
            'values': ['relu', 'gelu', 'silu', 'mish']
        },
        'filter_counts': {
            'values': [
                [16, 32, 64, 128, 256], 
                [32, 64, 128, 256, 512],
                [32, 32, 32, 32, 32],
                [64, 64, 64, 64, 64],
                [128, 128,128, 128,128],
                [128, 64, 32, 16, 8],
                [8, 16, 32, 64, 128]
            ]
        },
        'shape_of_filters_conv': {
            'values': [
                [(3, 3), (3, 3), (3, 3), (3, 3), (3, 3)],
                [(5, 5), (5, 5), (5, 5), (5, 5), (5, 5)].
                [(7, 7), (7, 7), (7, 7), (7, 7), (7, 7)].
                [(7, 7), (5, 5), (3, 3), (3, 3), (3, 3)],
                [(3, 3), (3, 3), (3, 3), (5, 5), (7, 7)],
                
            ]
        },
        'fc_layer': {
            'values': [128, 256, 512]
        },
        'batch_norm_use': {
            'values': [True, False]
        },
        'dropout': {
            'values': [0, 0.2, 0.3,0.5,0.8]
        },
        'data_aug': {
            'values': [True, False]
        }
    }
}
#######################################################
# Paart A Q1
# Define the CNN CLass  
class CNN(nn.Module):
    def __init__(self, num_classes=10, filter_counts=[32, 64, 128, 256, 512], filter_sizes=[(3, 3), (3, 3), (3, 3), (3, 3), (3, 3)],
                 activation_func=nn.ReLU(), fc_neurons=512, use_batch_norm=True,dropout_rate=0.2):
        super(CNNModel, self).__init__()
        #RGB channels
        in_channels = 3  
        # Store layers in List
        self.features = nn.ModuleList()
        
        # doing 5 conv-act-maxpool blocks
        for i in range(5):
            # Add  2D convolution layer
            conv = nn.Conv2d(in_channels, filter_counts[i], kernel_size=filter_sizes[i], padding='same')
            self.features.append(conv)
            
            # Add batch normalization
            if use_batch_norm:
                self.features.append(nn.BatchNorm2d(filter_counts[i]))
                
            # Add activation function
            self.features.append(activation_func)
            
            # Add max pooling layer
            self.features.append(nn.MaxPool2d(kernel_size=2, stride=2))
            
            # Update in_channels for the next layer
            in_channels = filter_counts[i]
            
        # Adaptive pooling to handle different input sizes
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Fully connected layers
        self.classifier = nn.Sequential(
            nn.Linear(filter_counts[-1], fc_neurons),activation_func,
            nn.Dropout(dropout_rate) if dropout_rate > 0 else nn.Identity(),
            nn.Linear(fc_neurons, num_classes)
        )

    def forward(self, x):
        # Pass input through feature layers
        for layer in self.features:
            x = layer(x)
            
        # Global average pooling
        x = self.adaptive_pool(x)
        
        # Flatten the tensor
        x = torch.flatten(x, 1)
        
        # Pass through classifier
        x = self.classifier(x)
        return x
        

########################################################

# Training function
def train_model(config=None):
    # Initialize wandb
    with wandb.init(config=config):
        # Get the configuration
        config = wandb.config
        
        # Set device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")
        
        # Set input size to smaller dimension to run cuda OOM
        #changes image size 160*160
        input_size = 160  
        
        # Set up data augmentation based on config
        if config.data_aug:
            train_transform = transforms.Compose([
                transforms.Resize((input_size, input_size)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(15),
                transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        else:
            train_transform = transforms.Compose([
                transforms.Resize((input_size, input_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        
        val_transform = transforms.Compose([
            transforms.Resize((input_size, input_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # Get data loaders
        train_loader, val_loader, classes = get_dataset_and_loaders(
            '/kaggle/input/inaturalist/inaturalist_12K',
            train_transform,
            val_transform,
            batch_size=config.train_batch_size
        )
        
        # Get activation function
        activation = get_activation(config.activation)
        
        # Create the model
        model = CNN(
            num_classes=len(classes),
            filter_counts=config.filter_counts,
            filter_sizes=config.shape_of_filters_conv,
            activation_func=activation,
            fc_neurons=config.fc_layer,
            use_batch_norm=config.batch_norm_use,
            dropout_rate=config.dropout
        )
        
        model.to(device)
        
        # Define loss function and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
        
        # Learning rate scheduler
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3, verbose=True)
        
        # Track best validation accuracy
        best_val_acc = 0.0
        
        # Training loop
        for epoch in range(config.epochs):
            # Training phase
            model.train()
            running_loss = 0.0
            correct = 0
            total = 0
            
            for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{config.epochs} [Train]"):
                inputs, labels = inputs.to(device), labels.to(device)
                
                # Zero the parameter gradients
                optimizer.zero_grad()
                
                # Forward pass
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                # Backward pass and optimize
                loss.backward()
                optimizer.step()
                
                # Statistics
                running_loss += loss.item() * inputs.size(0)
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
            
            train_loss = running_loss / len(train_loader.dataset)
            train_acc = correct / total
            
            # Validation phase
            model.eval()
            val_loss = 0.0
            correct = 0
            total = 0
            
            with torch.no_grad():
                for inputs, labels in tqdm(val_loader, desc=f"Epoch {epoch+1}/{config.epochs} [Val]"):
                    inputs, labels = inputs.to(device), labels.to(device)
                    
                    # Forward pass
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    
                    # Statistics
                    val_loss += loss.item() * inputs.size(0)
                    _, predicted = outputs.max(1)
                    total += labels.size(0)
                    correct += predicted.eq(labels).sum().item()
            
            val_loss = val_loss / len(val_loader.dataset)
            val_acc = correct / total
            
            # Log metrics
            wandb.log({
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "train_accuracy": train_acc,
                "val_loss": val_loss,
                "val_accuracy": val_acc
            })
            
            print(f"Epoch {epoch+1}/{config.epochs}, " f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, "
                  f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
                # Save best model
	    if val_acc > best_val_acc:
		best_val_acc = val_acc
		torch.save(model.state_dict(), 'best_model.pth')
		print(f"New best model saved with Val Acc: {val_acc:.4f}")
                  
                  
# Initialize wandb
wandb.login()

# Create the sweep
sweep_id = wandb.sweep(sweep_config, project='inaturalist-cnn')

# Run the sweep
wandb.agent(sweep_id, train_model, count=10) 

####################################################################################
## PartA Q 4
## Test data 

def test_loader(data_dir, test_transform, batch_size=64):
    test_dataset = ImageFolder(os.path.join(data_dir, 'val'), transform=test_transform)  # Adjust path if needed
    return DataLoader(test_dataset, batch_size, shuffle=False, num_workers=2)

# Load best model
model.load_state_dict(torch.load('best_model.pth'))
model.eval()

# Test data
test_loader = test_loader(data_dir, val_transform, best_config['train_batch_size'])

# Evaluation
correct, total = 0, 0
with torch.no_grad():
    for inputs, labels in tqdm(test_loader, desc="Testing"):
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

test_acc = correct / total
print(f"Test Accuracy: {test_acc:.4f}") 


# Ensure model is in eval mode for Test
model.eval()

# Get dataset and class info
test_dataset = test_loader.dataset
class_names = test_dataset.classes
num_classes = len(class_names)

# Collect indices of samples per class
class_indices = {i: [] for i in range(num_classes)}
for idx, (_, label) in enumerate(test_dataset.samples):
    class_indices[label].append(idx)

# Randomly pick 10 samples per class
samples_per_class = 10
selected_indices = []
for class_id in range(num_classes):
    if len(class_indices[class_id]) >= samples_per_class:
        selected_indices.extend(random.sample(class_indices[class_id], samples_per_class))

# Get selected samples
selected_images = []
true_labels = []
for idx in selected_indices:
    img, label = test_dataset[idx]
    selected_images.append(img)
    true_labels.append(label)

# Convert to tensor batch
batch = torch.stack(selected_images).to(device)
true_labels = torch.tensor(true_labels).to(device)

# Get predictions
with torch.no_grad():
    outputs = model(batch)
_, pred_labels = outputs.max(1)

# Denormalize images (adjust mean/std if needed)
mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(device)
std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(device)
denormalized = batch * std + mean
denormalized = torch.clamp(denormalized, 0, 1)





plt.figure(figsize=(20, 5 * num_classes))  # Adjust height dynamically
for class_id in range(num_classes):
    class_samples = [i for i, lbl in enumerate(true_labels) if lbl == class_id]
    for j, sample_idx in enumerate(class_samples[:3]):  # Show first 3 samples per class
        plt.subplot(num_classes, 10, class_id * 10 + j + 1)
        img = denormalized[sample_idx].cpu().numpy().transpose((1, 2, 0))
        plt.imshow(img)
        plt.axis('off')
        
        true_name = class_names[true_labels[sample_idx]]
        pred_name = class_names[pred_labels[sample_idx]]
        color = "green" if true_labels[sample_idx] == pred_labels[sample_idx] else "red"
        plt.title(f"T: {true_name}\nP: {pred_name}", color=color, fontsize=8)

plt.tight_layout()
plt.show()

# Save the figure to a temporary buffer (no physical file)
buf = io.BytesIO()
plt.savefig(buf, format='png', bbox_inches='tight', dpi=150)
buf.seek(0)
img = Image.open(buf)

# Log to wandb
wandb.log({
    "Predictions (3 per class)": wandb.Image(img, caption="3 samples per class")
})

# Close the figure to free memory
plt.close(fig)
buf.close()
wandb.finish()
#######################################################################

