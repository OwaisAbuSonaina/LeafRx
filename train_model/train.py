# train.py

# Setup
from torch import nn
import torch
from torchvision import models

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"The device we are currently on is: {device}")

import engine
from prepare_dataset import train_loader, val_loader
from utils import set_seeds, save_model


# Import the Model
model_weights = models.ConvNet
model = models.convnext_tiny(weights=model_weights).to(device)

# Freeze all base (feature extractor) layers
for param in model.parameters():
    param.requires_grad = False

# Replace the classifier head
num_classes = 38  # PlantVillage has 38 classes
model.classifier[2] = nn.Linear(model.classifier[2].in_features, num_classes)

# Define loss function and optimizer
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(params=model.classifier[2].parameters(), lr=0.001)

# Train the Model
set_seeds()
results = engine.train(
    model=model,
    train_dataloader=train_loader,
    test_dataloader=val_loader,
    optimizer=optimizer,
    loss_fn=loss_fn,
    epochs=5,
    device=device,
)

# Save the Model
save_model(model=model, target_dir="./models", model_name="ConvNext-Tiny_First_Try.pth")