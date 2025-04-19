import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
import wandb
from tqdm import tqdm

def get_activation(name):
    activations = {
        'relu': nn.ReLU(),
        'gelu': nn.GELU(),
        'silu': nn.SiLU(),
        'mish': nn.Mish()
    }
    return activations.get(name, nn.ReLU())

def get_dataset_and_loaders(data_dir, train_transform, val_transform, batch_size=64, val_split=0.2):
    full_train_dataset = ImageFolder(os.path.join(data_dir, 'train'), transform=train_transform)
    targets = np.array(full_train_dataset.targets)
    train_indices, val_indices = [], []
    for class_idx in range(len(full_train_dataset.classes)):
        class_indices = np.where(targets == class_idx)[0]
        n_val = int(len(class_indices) * val_split)
        np.random.shuffle(class_indices)
        val_indices.extend(class_indices[:n_val])
        train_indices.extend(class_indices[n_val:])
    train_dataset = Subset(full_train_dataset, train_indices)
    val_dataset = Subset(full_train_dataset, val_indices)
    val_dataset.dataset.transform = val_transform
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    return train_loader, val_loader, full_train_dataset.classes

class CNN(nn.Module):
    def __init__(self, num_classes=10, filter_counts=[32, 32, 32, 32, 32], filter_sizes=[(3,3)]*5,
                 activation_func=nn.ReLU(), fc_neurons=512, use_batch_norm=True, dropout_rate=0.2):
        super(CNN, self).__init__()
        self.features = nn.ModuleList()
        in_channels = 3
        for i in range(5):
            self.features.append(nn.Conv2d(in_channels, filter_counts[i], kernel_size=filter_sizes[i], padding='same'))
            if use_batch_norm:
                self.features.append(nn.BatchNorm2d(filter_counts[i]))
            self.features.append(activation_func)
            self.features.append(nn.MaxPool2d(kernel_size=2, stride=2))
            in_channels = filter_counts[i]
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Linear(filter_counts[-1], fc_neurons),
            activation_func,
            nn.Dropout(dropout_rate) if dropout_rate > 0 else nn.Identity(),
            nn.Linear(fc_neurons, num_classes)
    def forward(self, x):
        for layer in self.features:
            x = layer(x)
        x = self.adaptive_pool(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)
        


def train(config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_size = 160
    train_transform = transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]) if config['data_aug'] else transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    val_transform = transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    train_loader, val_loader, classes = get_dataset_and_loaders(
        config['base_dir'], train_transform, val_transform, config['train_batch_size'])
    model = CNN(
        num_classes=len(classes),
        filter_counts=config['filter_counts'],
        filter_sizes=config['shape_of_filters_conv'],
        activation_func=get_activation(config['activation']),
        fc_neurons=config['fc_layer'],
        use_batch_norm=config['batch_norm_use'],
        dropout_rate=config['dropout']
    ).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.1, patience=3)
    best_val_acc = 0.0
    for epoch in range(config['epochs']):
        model.train()
        train_loss, train_correct, train_total = 0.0, 0, 0
        for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1} [Train]"):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            train_total += labels.size(0)
            train_correct += predicted.eq(labels).sum().item()
        train_loss /= len(train_loader.dataset)
        train_acc = train_correct / train_total
        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        with torch.no_grad():
            for inputs, labels in tqdm(val_loader, desc=f"Epoch {epoch+1} [Val]"):
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                val_loss += criterion(outputs, labels).item() * inputs.size(0)
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()
        val_loss /= len(val_loader.dataset)
        val_acc = val_correct / val_total
        wandb.log({
            "epoch": epoch+1,
            "train_loss": train_loss,
            "train_accuracy": train_acc,
            "val_loss": val_loss,
            "val_accuracy": val_acc
        })
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), 'best_model.pth')
        print(f"Epoch {epoch+1}: Train Loss {train_loss:.4f}, Acc {train_acc:.4f}; Val Loss {val_loss:.4f}, Acc {val_acc:.4f}")

sweep_config = {
    'method': 'random',
    'parameters': {
        'learning_rate': {'values': [0.001, 0.0005, 0.0001, 0.01]},
        'train_batch_size': {'values': [64, 128]},
        'epochs': {'values': [10, 20]},
        'activation': {'values': ['relu', 'gelu', 'silu', 'mish']},
        'filter_counts': {'values': [
            [16, 32, 64, 128, 256], [32, 64, 128, 256, 512],
            [32]*5, [64]*5, [128]*5, [128, 64, 32, 16, 8], [8, 16, 32, 64, 128]]},
        'shape_of_filters_conv': {'values': [
            [(3,3)]*5, [(5,5)]*5, [(7,7)]*5,
            [(7,7), (5,5), (3,3), (3,3), (3,3)],
            [(3,3)]*3 + [(5,5), (7,7)]]},
        'fc_layer': {'values': [128, 256, 512]},
        'batch_norm_use': {'values': [True, False]},
        'dropout': {'values': [0, 0.2, 0.3, 0.5, 0.8]},
        'data_aug': {'values': [True, False]}
    }
}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--wandb_entity", default="manglesh-patidar-cs24m025")
    parser.add_argument("--wandb_project", default="inaturalist-cnn")
    parser.add_argument("--sweep", action="store_true")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--learning_rate", type=float, default=0.001)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--activation", default="relu")
    parser.add_argument("--num_filters", type=int, nargs=5, default=[32]*5)
    parser.add_argument("--filter_sizes", type=int, nargs=5, default=[3]*5)
    parser.add_argument("--batch_norm", default="true")
    parser.add_argument("--dense_layer", type=int, default=128)
    parser.add_argument("--augmentation", default="yes")
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--base_dir", default="inaturalist_12K")
    args = parser.parse_args()
    
    if args.sweep:
        sweep_id = wandb.sweep(sweep_config, project=args.wandb_project, entity=args.wandb_entity)
        wandb.agent(sweep_id, lambda: train(wandb.config), count=20)
    else:
        config = {
            'learning_rate': args.learning_rate,
            'train_batch_size': args.batch_size,
            'epochs': args.epochs,
            'activation': args.activation,
            'filter_counts': args.num_filters,
            'shape_of_filters_conv': [(s,s) for s in args.filter_sizes],
            'fc_layer': args.dense_layer,
            'batch_norm_use': args.batch_norm.lower() == "true",
            'dropout': args.dropout,
            'data_aug': args.augmentation.lower() == "yes",
            'base_dir': args.base_dir,
            'weight_decay': args.weight_decay
        }
        wandb.init(project=args.wandb_project, entity=args.wandb_entity, config=config)
        train(config)
        wandb.finish()

if __name__ == "__main__":
    main()

