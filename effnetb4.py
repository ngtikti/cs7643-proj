import math
import time
from PIL import Image
import io
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import csv 
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models, datasets
import torchvision.transforms.functional as TF
from sklearn.model_selection import train_test_split
from torch.utils.data.sampler import SubsetRandomSampler
import random

import os
import shutil
import gc


def set_seed(seed: int):
    random.seed(seed)                      # Python's random module
    np.random.seed(seed)                  # NumPy
    torch.manual_seed(seed)               # PyTorch CPU
    torch.cuda.manual_seed(seed)          # PyTorch GPU
    #torch.cuda.manual_seed_all(seed)      # If using multi-GPU
    #torch.backends.cudnn.deterministic = True  # Ensures deterministic behavior
    #torch.backends.cudnn.benchmark = False     # Disables performance optimization for reproducibility

# Example usage
set_seed(42)

# Check device availability
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("You are using device: %s" % device)


torch.cuda.empty_cache() 
gc.collect() 


# testing for b7 model
############### hyperparams ##################
valid_size = 0.15
batch_size = 64 #to keep it comparable for training times with ViT
learning_rate = 1e-4
weight_decay=1e-4
epochs = 10

net_weights='IMAGENET1K_V1' #'IMAGENET1K_V1' # None #'DEFAULT' or weights='IMAGENET1K_V1' or 'IMAGENET1K_V2'

label_to_process = "race"
input_size = 150 # standard resolution for all nets

###############################################
image_folder = "UTKFace"  # Replace with the path to your folder with images
train_folder = "seg_train"
test_folder = "seg_test"

train_image_path = train_folder + '/' + label_to_process
test_image_path = test_folder + '/' + label_to_process

#transform to input size for model
transform = transforms.Compose([
    transforms.Resize((input_size), interpolation=transforms.InterpolationMode.BILINEAR), #TODO: if got time, can explore testing different interpolation for scaling up
    transforms.ToTensor()
])

test_data = datasets.ImageFolder(test_image_path, transform=transform)
train_data = datasets.ImageFolder(train_image_path, transform=transform)

# percentage of training set to use as validation
num_classes = len(train_data.classes)
print(f'number of classes for {label_to_process}: {num_classes}')

# get training indices that wil be used for validation
train_size = len(train_data)
indices = list(range(train_size))
np.random.shuffle(indices)
split = int(np.floor(valid_size * train_size))
train_idx, valid_idx = indices[split:], indices[:split]

# define samplers to obtain training and validation batches
train_sampler = SubsetRandomSampler(train_idx)
valid_sampler = SubsetRandomSampler(valid_idx)

# prepare data loaders
train_loader = DataLoader(train_data, batch_size=batch_size, sampler=train_sampler)
val_loader = DataLoader(train_data, batch_size=batch_size, sampler=valid_sampler)
test_loader = DataLoader(test_data, batch_size=batch_size)

print (f"Original train data = {train_size}, \nAfter split : {train_size-split} & {split}")
print (f'Shape = {train_data[0][0].shape}') 


model_b4 = models.efficientnet_b4(weights = net_weights) #weights='DEFAULT' or weights='IMAGENET1K_V1
print(model_b4)

#updating classifier, with kevin's structure
model_b4.classifier[1] = nn.Sequential(
    nn.Dropout(0.4),
    nn.BatchNorm1d(1792),  # EfficientNet B7 has 1280 features before the classifier
    nn.ReLU(),
    nn.Linear(1792, 256),
    nn.ReLU(),
    nn.Linear(256, 256),
    nn.Dropout(0.5),
    nn.Linear(256, num_classes)  # Output layer with num_classes
)

model = model_b4.to(device)

# Count total parameters
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f"Total parameters: {total_params}")
print(f"Trainable parameters: {trainable_params}")

optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
criterion = nn.CrossEntropyLoss()


############## Training ################
# Initialize lists to store metrics
train_losses = []
val_losses = []
train_accuracies = []
val_accuracies = []

# Training loop
start_time = time.time()
for epoch in range(epochs):
    start_time_epoch = time.time()
    print(f'training epoch: {epoch+1}')
    model.train()  # Set the model to training mode
    running_loss = 0.0
    correct_preds = 0
    total_preds = 0
    count = 0
    for inputs, labels in train_loader:
        count+=1
        if count%50 == 0:
            print(f'batch {count} of {len(train_loader)}')
        inputs = inputs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()  # Zero the gradients
        outputs = model(inputs)  # Forward pass
        loss = criterion(outputs, labels)  # Compute loss
        loss.backward()  # Backward pass
        optimizer.step()  # Update weights
        
        running_loss += loss.item()
        _, predicted = torch.max(outputs, 1)  # Get predicted labels
        correct_preds += (predicted == labels).sum().item()
        total_preds += labels.size(0)
    
    # Calculate training accuracy and loss
    train_loss = running_loss / len(train_loader)
    train_acc = correct_preds / total_preds
    
    train_losses.append(train_loss)
    train_accuracies.append(train_acc)
    
    # Validation loop
    model.eval()  # Set the model to evaluation mode
    val_loss = 0.0
    val_correct_preds = 0
    val_total_preds = 0
    
    with torch.no_grad():  # No need to compute gradients during validation
        for inputs, labels in val_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            val_correct_preds += (predicted == labels).sum().item()
            val_total_preds += labels.size(0)
    
    # Calculate validation accuracy and loss
    val_loss = val_loss / len(val_loader)
    val_acc = val_correct_preds / val_total_preds
    
    val_losses.append(val_loss)
    val_accuracies.append(val_acc)
    
    # Print stats for each epoch
    print(f"Epoch {epoch + 1}/{epochs}")
    print(f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_acc:.4f}")
    print(f"Val Loss: {val_loss:.4f}, Val Accuracy: {val_acc:.4f}")
    end_time_epoch = time.time()
    total_time_epoch = end_time_epoch - start_time_epoch
    print(f"Time taken for {epoch} epoch: {total_time_epoch:.2f} seconds")


end_time = time.time()

# Calculate elapsed time
total_time = end_time - start_time
print(f"Total training time for {epochs} epochs: {total_time:.2f} seconds, {total_time/60} mins")


# Final evaluation on the test data
model.eval()
test_loss = 0.0
test_correct_preds = 0
test_total_preds = 0

with torch.no_grad():
    for inputs, labels in test_loader:
        inputs = inputs.to(device)
        labels = labels.to(device)
        
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        test_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        test_correct_preds += (predicted == labels).sum().item()
        test_total_preds += labels.size(0)

test_loss = test_loss / len(test_loader)
test_acc = test_correct_preds / test_total_preds
print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.4f}")


######### Plotting ###########
os.makedirs("results", exist_ok=True)

# Plot Loss
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(range(1, epochs +1), train_losses, label="Training Loss")
plt.plot(range(1, epochs +1), val_losses, label="Validation Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Training and Validation Loss")
plt.legend()

# Plot Accuracy
plt.subplot(1, 2, 2)
plt.plot(range(1, epochs +1), train_accuracies, label="Training Accuracy")
plt.plot(range(1, epochs +1), val_accuracies, label="Validation Accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy (%)")
plt.title("Training and Validation Accuracy")
plt.legend()

# Save the plot to /results/ directory
plt.tight_layout()
output_path = f'results/training_validation_curves_b4_{str(net_weights)}_{label_to_process}_{epochs}.png'
plt.savefig(output_path, dpi=300)
print(f"Plot saved to {output_path}")

plt.close()







