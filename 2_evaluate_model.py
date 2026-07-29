"""
Satellite Image Classifier - Evaluation Script
Tests the trained model and generates confusion matrix
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms, models
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import numpy as np

# ============================================
# 1. SETUP
# ============================================

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
data_path = Path('data/EuroSAT_RGB')
model_path = Path('models/resnet50_satellite_classifier.pth')

if not model_path.exists():
    print(f"❌ ERROR: Model not found at {model_path}")
    print("Did you run 1_train_classifier.py first?")
    exit()

# ============================================
# 2. LOAD DATA (SAME AS TRAINING)
# ============================================

test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

full_dataset = datasets.ImageFolder(root=str(data_path), transform=test_transform)
class_names = full_dataset.classes

# Get test set (same split as training)
train_size = int(0.8 * len(full_dataset))
val_size = int(0.1 * len(full_dataset))
test_size = len(full_dataset) - train_size - val_size

train_set, val_set, test_set = random_split(
    full_dataset, 
    [train_size, val_size, test_size]
)

test_loader = DataLoader(test_set, batch_size=32, shuffle=False)

# ============================================
# 3. LOAD MODEL
# ============================================

model = models.resnet50(pretrained=True)
for param in model.parameters():
    param.requires_grad = False

num_features = model.fc.in_features
model.fc = nn.Linear(num_features, len(class_names))
model.load_state_dict(torch.load(model_path, map_location=device))
model = model.to(device)
model.eval()

print("✓ Model loaded")

# ============================================
# 4. EVALUATE ON TEST SET
# ============================================

print("\nEvaluating on test set...")

all_preds = []
all_labels = []

with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.numpy())

# ============================================
# 5. PRINT RESULTS
# ============================================

print(f"\n{'='*60}")
print("CLASSIFICATION REPORT (Per-Class Performance)")
print(f"{'='*60}")

print(classification_report(all_labels, all_preds, target_names=class_names))

# Overall accuracy
overall_accuracy = sum(np.array(all_preds) == np.array(all_labels)) / len(all_labels)
print(f"Overall Test Accuracy: {overall_accuracy:.4f} ({overall_accuracy*100:.2f}%)")

# ============================================
# 6. GENERATE CONFUSION MATRIX
# ============================================

print("\nGenerating confusion matrix...")

cm = confusion_matrix(all_labels, all_preds)

plt.figure(figsize=(12, 10))
sns.heatmap(
    cm, 
    annot=True, 
    fmt='d', 
    xticklabels=class_names, 
    yticklabels=class_names,
    cmap='Blues',
    cbar_kws={'label': 'Count'}
)
plt.xlabel('Predicted Class', fontsize=12)
plt.ylabel('Actual Class', fontsize=12)
plt.title('Confusion Matrix - Satellite Image Classifier', fontsize=14, fontweight='bold')
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()

# Save the figure
results_path = Path('results')
results_path.mkdir(exist_ok=True)
save_path = results_path / 'confusion_matrix.png'
plt.savefig(save_path, dpi=300, bbox_inches='tight')
print(f"✓ Confusion matrix saved to: {save_path}")

plt.show()

print("\n✓ Evaluation complete!")