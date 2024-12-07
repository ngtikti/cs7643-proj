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

"""
To process UTKFace into proper folder for torch Dataset format
"""

image_folder = "UTKFace"  # Replace with the path to your folder with images
train_folder = "seg_train"
test_folder = "seg_test"

# Create destination folders
os.makedirs(train_folder, exist_ok=True)
os.makedirs(test_folder, exist_ok=True)

label_mapping = {"age" : 0, "gender" : 1, "race" : 2}
label_to_process = 'race' #'gender' 'race' 'age' #<< change this for the analysis to change


# Get a list of all image files
image_files = [f for f in os.listdir(image_folder) if f.endswith(".jpg")]

# Split the data into training and testing sets (80% train, 20% test)
train_files, test_files = train_test_split(image_files, test_size=0.13, random_state=42)

# Move training files in format "seg_train/label_to_predict/class_of_interest
#processing for age
for file_name in train_files:
    try:
        split = file_name.split('_')
        class_to_input = str(int(split[label_mapping[label_to_process]]))
        path = train_folder + '/' + label_to_process + '/' + class_to_input
        os.makedirs(path, exist_ok=True)
        #copy into following 
        src_path = os.path.join(image_folder, file_name)
        dst_path = os.path.join(path, file_name)
        shutil.copy(src_path, dst_path)
    except Exception as ex:
        print(f'exception: {str(ex)}')
        continue

# Move testing files
for file_name in test_files:
    try:
        split = file_name.split('_')
        class_to_input = str(int(split[label_mapping[label_to_process]]))
        path = test_folder + '/' + label_to_process + '/' + class_to_input
        os.makedirs(path, exist_ok=True)
        #copy into following 
        src_path = os.path.join(image_folder, file_name)
        dst_path = os.path.join(path, file_name)
        shutil.copy(src_path, dst_path)
    except Exception as ex:
        print(f'exception: {str(ex)}')
        continue

print('done')

train_image_path = train_folder + '/' + label_to_process
test_image_path = test_folder + '/' + label_to_process

print(f"Training files moved to: {train_image_path}")
print(f"Testing files moved to: {test_image_path}")
batch_size = 32
transform = transforms.Compose([transforms.Resize(150),
                                transforms.ToTensor()])

test_data = datasets.ImageFolder(test_image_path, transform=transform)
train_data = datasets.ImageFolder(train_image_path, transform=transform)
train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True) # TODO: use the ImageFolder dataset to create the DataLoader
test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=True) 
print (type (train_data))


print (f'Length of Train data & Test data = {len (train_data)} & {len (test_data)}' )
print (f'Shape = {train_data[0][0].shape}')
