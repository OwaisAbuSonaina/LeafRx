# prepare_dataset.py

# Setup 
import torch
import torchvision
from torchvision import transforms, datasets, transforms
import numpy as np
from sklearn.model_selection import train_test_split
import os, shutil, glob

print(f"torch version: {torch.__version__}")
print(f"torchvision version: {torchvision.__version__}")


# Splitting the Dataset
source_dir = "./plantvillage dataset/color"
dest_dir = "./plantvillage dataset/color_splitted"

classes = os.listdir(source_dir)
for cls in classes:
    imgs = glob.glob(os.path.join(source_dir, cls, "*.jpg"))
    train, temp = train_test_split(imgs, test_size=0.3, random_state=42)
    val, test = train_test_split(temp, test_size=0.5, random_state=42)

    for name, split in zip([train, val, test], ["train", "val", "test"]):
        os.makedirs(os.path.join(dest_dir, split, cls), exist_ok=True)
        for img in name:
            shutil.copy(img, os.path.join(dest_dir, split, cls))

# Prepare Transforms
transform = {
    "train": transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    ),
    "val": transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    ),
}

# Create torch Datasets
data_dir = "./plantvillage dataset/color_splitted"
train_dataset = datasets.ImageFolder(
    os.path.join(data_dir, "train"), transform=transform["train"]
)
val_dataset = datasets.ImageFolder(
    os.path.join(data_dir, "val"), transform=transform["val"]
)

# Applying WieghtedRandomSampler ONLY on the training set
from torch.utils.data import WeightedRandomSampler

targets = [label for _, label in train_dataset.samples]
class_counts = np.bincount(targets)
class_weights = 1.0 / class_counts
sample_weights = [class_weights[t] for t in targets]
sampler = WeightedRandomSampler(
    sample_weights, num_samples=len(sample_weights), replacement=True
)

# Create DataLoaders
from torch.utils.data import DataLoader

train_loader = DataLoader(train_dataset, batch_size=16, sampler=sampler)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=True)