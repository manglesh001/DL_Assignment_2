import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, Subset
from torchvision import models
import numpy as np
from tqdm import tqdm
import os
import wandb
import argparse
import itertools

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
def get_dataset_and_loaders(data_dir, train_transform, val_transform, batch_size=32, val_split=0.2):
    
    # Load full dataset from train directory
    full_dataset = ImageFolder(os.path.join(data_dir, 'train'), transform=train_transform)
    
    # Create stratified train/validation split
    targets = np.array(full_dataset.targets)  
    train_indices, val_indices = [], [] 
    
    # For each class, split indices while maintaining class distribution
    for class_idx in np.unique(targets):
        class_indices = np.where(targets == class_idx)[0] 
        n_val = int(len(class_indices) * val_split) 
        np.random.shuffle(class_indices)  
        
        val_indices.extend(class_indices[:n_val])
        train_indices.extend(class_indices[n_val:])
    
    # Create subset datasets
    train_dataset = Subset(full_dataset, train_indices)
    val_dataset = Subset(full_dataset, val_indices)
    val_dataset.dataset.transform = val_transform
    
    return (
        DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4),
        DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4),
        full_dataset.classes  
    )

def create_model(strategy, num_classes, freeze_k=5, dropout_rate=0.5, hidden_units=512):

    # Load pre-trained ResNet50 with ImageNet weights
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
    
    if strategy == 'last_layer':
        # Freeze all layers except the final classifier
        for param in model.parameters():
            param.requires_grad = False
            
        # Replace final fully connected layer with custom classifier
        num_features = model.fc.in_features 
        model.fc = nn.Sequential(
            nn.Dropout(dropout_rate), 
            nn.Linear(num_features, hidden_units), 
            nn.ReLU(),  
            nn.Linear(hidden_units, num_classes)  
        )
        # Unfreeze the new classifier layers
        for param in model.fc.parameters():
            param.requires_grad = True
            
    elif strategy == 'freeze_k':
        # Freeze first k layers of the network
        # List all major components of ResNet50
        layers = [
            model.conv1, model.bn1, model.relu, model.maxpool,
            model.layer1, model.layer2, model.layer3, model.layer4
        ]
        # Freeze parameters in first k layers
        for layer in layers[:freeze_k]:
            for param in layer.parameters():
                param.requires_grad = False
                
        # Replace final classifier similar to last_layer strategy
        model.fc = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(2048, hidden_units),
            nn.ReLU(),
            nn.Linear(hidden_units, num_classes)
        )
        
    elif strategy == 'full_model':
        # Train entire network (no freezing)
        # Still replace final classifier for our task
        model.fc = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(2048, hidden_units),
            nn.ReLU(),
            nn.Linear(hidden_units, num_classes))
        
    return model.to(device)  


def train_model(args):

    # Initialize Weights & Biases experiment tracking
    wandb.init(project=args.wandb_project, entity=args.wandb_entity, config=vars(args))
    
    # Define training transformations with data augmentation
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224),  
        transforms.RandomHorizontalFlip(),  
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),  
        transforms.ToTensor(),  
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  
    ])

    # Validation transformations 
    val_transform = transforms.Compose([
        transforms.Resize(256),  
        transforms.CenterCrop(224), 
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Get data loaders and class names
    train_loader, val_loader, classes = get_dataset_and_loaders(
        args.base_dir,
        train_transform,
        val_transform,
        batch_size=args.batch_size,
        val_split=0.2
    )

    # Initialize model with specified strategy
    model = create_model(
        strategy=args.strategy,
        num_classes=len(classes),
        freeze_k=args.freeze_k,
        dropout_rate=args.dropout_rate,
        hidden_units=args.classifier_hidden_units
    )

    # Configure optimizer parameters
    if args.strategy == 'last_layer':
        # Only optimize the classifier parameters
        params = model.fc.parameters()
    else:
        # Optimize all parameters that require gradients
        params = filter(lambda p: p.requires_grad, model.parameters())
    
    # Dictionary mapping optimizer names to classes
    optimizers = {
        'adam': optim.Adam,
        'nadam': optim.NAdam,
        'rmsprop': optim.RMSprop,
        'adamw': optim.AdamW
    }
    # Initialize selected optimizer
    optimizer = optimizers[args.optimizer.lower()](
        params, 
        lr=args.learning_rate,
        weight_decay=args.weight_decay
    )

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='max',  
        factor=args.scheduler_factor,  
        patience=args.scheduler_patience  
    )

    criterion = nn.CrossEntropyLoss()
    wandb.watch(model, log_freq=100)

    # Training loop
    best_val_acc = 0.0  
    for epoch in range(args.epochs):
        model.train() 
        train_loss, correct, total = 0.0, 0, 0  
        
        for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1} [Train]"):
            inputs, labels = inputs.to(device), labels.to(device)  
            
            optimizer.zero_grad()  
            outputs = model(inputs) 
            loss = criterion(outputs, labels) 
            loss.backward()  
            optimizer.step()  
            
            # Update training metrics
            train_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1) 
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item() 
        
        # Validation phase
        model.eval()  
        val_loss, val_correct, val_total = 0.0, 0, 0
        with torch.no_grad(): 
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                # Update validation metrics
                val_loss += loss.item() * inputs.size(0)
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()
        
        # Calculate epoch metrics
        train_loss /= len(train_loader.dataset) 
        train_acc = correct / total 
        val_loss /= len(val_loader.dataset)  
        val_acc = val_correct / val_total 
   
        # Log metrics to wandb
        wandb.log({
            "epoch": epoch,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "lr": optimizer.param_groups[0]['lr'] 
        })
        
        # Save best model checkpoint
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), 'best_model.pth')  
            wandb.save('best_model.pth') 
            
    wandb.finish() 

if __name__ == "__main__":
    wandb.login(key="6001619563748a57b4114b0bb090fd4129ba6122")

    parser = argparse.ArgumentParser(description="Fine-tune ResNet50 on iNaturalist dataset")
    
    parser.add_argument("--wandb_entity", "-we", default="manglesh-patidar-cs24m025",
                      help="Weights & Biases entity name")
    parser.add_argument("--wandb_project", "-wp", default="inaturalist-classification",
                      help="Weights & Biases project name")
    
    parser.add_argument("--strategy", "-s", 
                       choices=['last_layer', 'freeze_k', 'full_model'],
                       default='last_layer',
                       help="Fine-tuning strategy to use")
    parser.add_argument("--freeze_k", "-k", type=int, default=5,
                       help="Number of initial layers to freeze for freeze_k strategy")
    
    parser.add_argument("--dropout_rate", type=float, default=0.5,
                       help="Dropout rate for classifier")
    parser.add_argument("--classifier_hidden_units", type=int, default=512,
                       help="Number of units in hidden layer of classifier")
    parser.add_argument("--epochs", "-e", type=int, default=10,
                       help="Number of training epochs")
    parser.add_argument("--batch_size", "-b", type=int, default=16,
                       help="Batch size for training")
    parser.add_argument("--optimizer", "-o", 
                       choices=['adam', 'nadam', 'rmsprop', 'adamw'],
                       default='adamw',
                       help="Optimizer to use")
    parser.add_argument("--learning_rate", "-lr", type=float, default=1e-3,
                       help="Initial learning rate")
    parser.add_argument("--weight_decay", "-w_d", type=float, default=1e-4,
                       help="Weight decay for optimizer")
  
    
    # Data parameters
    parser.add_argument("--base_dir", "-br", default="inaturalist_12K",
                       help="Base directory containing dataset")
    
    args = parser.parse_args()  
    
    args.optimizer = args.optimizer.lower()
    
    wandb.login()
    train_model(args)
