"""
Satellite Image Classifier - Training Script
Trains ResNet50 on EuroSAT satellite images
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms, models
import os
from pathlib import Path

# ============================================
# 1. SETUP
# ============================================

# Check if GPU is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Path to your EuroSAT dataset
data_path = Path('data/EuroSAT_RGB')

# Check if dataset exists
if not data_path.exists():
    print(f"❌ ERROR: Dataset not found at {data_path}")
    print(f"Make sure you downloaded EuroSAT and extracted it to: {data_path}")
    exit()

# ============================================
# 2. DATA PREPROCESSING
# ============================================

# Define image transforms
# Training transforms include augmentation (random flips)
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# Test/validation transforms - no augmentation
test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# Load all images with their labels
print("Loading dataset...")
full_dataset = datasets.ImageFolder(root=str(data_path), transform=train_transform)
num_classes = len(full_dataset.classes)
class_names = full_dataset.classes

print(f"✓ Dataset loaded: {len(full_dataset)} images")
print(f"✓ Classes: {class_names}")

# Split into train (80%), validation (10%), test (10%)
train_size = int(0.8 * len(full_dataset))
val_size = int(0.1 * len(full_dataset))
test_size = len(full_dataset) - train_size - val_size

train_set, val_set, test_set = random_split(
    full_dataset, 
    [train_size, val_size, test_size]
)

print(f"✓ Train set: {len(train_set)} images")
print(f"✓ Validation set: {len(val_set)} images")
print(f"✓ Test set: {len(test_set)} images")

# Create DataLoaders (feeds images in batches)
train_loader = DataLoader(train_set, batch_size=32, shuffle=True)
val_loader = DataLoader(val_set, batch_size=32, shuffle=False)
test_loader = DataLoader(test_set, batch_size=32, shuffle=False)

# ============================================
# 3. BUILD THE MODEL
# ============================================

print("\nBuilding model...")

# Load pretrained ResNet50
model = models.resnet50(pretrained=True)

# Freeze all layers (don't train them)
for param in model.parameters():
    param.requires_grad = False

# Replace final layer with new one for 10 classes
num_features = model.fc.in_features
model.fc = nn.Linear(num_features, num_classes)

# Move to GPU if available
model = model.to(device)

print(f"✓ Model loaded and moved to {device}")

# ============================================
# 4. TRAINING SETUP
# ============================================

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.fc.parameters(), lr=0.001)
num_epochs = 10

print(f"\n{'='*50}")
print("STARTING TRAINING")
print(f"{'='*50}")

# ============================================
# 5. TRAINING LOOP
# ============================================

for epoch in range(num_epochs):
    # TRAINING PHASE
    model.train()
    running_loss = 0.0
    
    for batch_idx, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)
        
        # Forward pass
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        
        # Print progress every 20 batches
        if (batch_idx + 1) % 20 == 0:
            print(f"  Epoch {epoch+1}/{num_epochs}, Batch {batch_idx+1}/{len(train_loader)}, Loss: {loss.item():.4f}")
    
    avg_train_loss = running_loss / len(train_loader)
    
    # VALIDATION PHASE
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    val_accuracy = correct / total
    
    print(f"Epoch {epoch+1}/{num_epochs} - Loss: {avg_train_loss:.4f}, Val Accuracy: {val_accuracy:.4f}\n")

# ============================================
# 6. SAVE THE MODEL
# ============================================

models_path = Path('models')
models_path.mkdir(exist_ok=True)

model_path = models_path / 'resnet50_satellite_classifier.pth'
torch.save(model.state_dict(), model_path)

print(f"\n✓ Model saved to: {model_path}")
print("\nTraining complete! Next, run: python 2_evaluate_model.py")