"""
Deforestation Detection Script
Compares satellite images from two time periods using the trained classifier
Detects where Forest has been converted to other land types
"""

import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import pickle

# ============================================
# SETUP
# ============================================

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model_path = Path('models/resnet50_satellite_classifier.pth')
data_path = Path('data/sentinel_images')
results_path = Path('results')
results_path.mkdir(exist_ok=True)

print(f"Using device: {device}")

# Check if model exists
if not model_path.exists():
    print(f"❌ ERROR: Model not found at {model_path}")
    print("Did you run 1_train_classifier.py first?")
    exit()

# ============================================
# LOAD MODEL
# ============================================

print("Loading trained model...")

model = models.resnet50(pretrained=True)
num_features = model.fc.in_features
model.fc = nn.Linear(num_features, 10)
model.load_state_dict(torch.load(model_path, map_location=device))
model = model.to(device)
model.eval()

print("✓ Model loaded")

# Image processing pipeline
test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

class_names = [
    'AnnualCrop', 'Forest', 'HerbaceousVegetation', 'Highway',
    'Industrial', 'Pasture', 'PermanentCrop', 'Residential', 'River', 'SeaLake'
]

# ============================================
# FUNCTIONS
# ============================================

def chop_image_into_patches(image_path, patch_size=64):
    """
    Split a satellite image into 64x64 patches
    Returns: list of patches, list of positions, original image size
    """
    if not Path(image_path).exists():
        raise FileNotFoundError(f"Image not found: {image_path}")
    
    img = Image.open(image_path).convert('RGB')
    width, height = img.size
    patches = []
    positions = []
    
    for i in range(0, width - patch_size, patch_size):
        for j in range(0, height - patch_size, patch_size):
            patch = img.crop((i, j, i + patch_size, j + patch_size))
            patches.append(patch)
            positions.append((i, j))
    
    return patches, positions, img.size

def create_landcover_map(image_path, year_label=""):
    """
    Classify all patches and create a land cover map
    Returns: dictionary of classifications by position
    """
    print(f"\nProcessing {year_label}: {image_path}")
    
    try:
        patches, positions, img_size = chop_image_into_patches(image_path)
    except FileNotFoundError:
        print(f"❌ Error: Image file not found at {image_path}")
        return None
    
    landcover_map = {}
    
    for idx, (patch, pos) in enumerate(zip(patches, positions)):
        patch_tensor = test_transform(patch).unsqueeze(0).to(device)
        
        with torch.no_grad():
            output = model(patch_tensor)
            probabilities = torch.nn.functional.softmax(output, dim=1)
            confidence, predicted = torch.max(probabilities, 1)
        
        landcover_map[pos] = {
            'class': class_names[predicted.item()],
            'confidence': confidence.item()
        }
        
        if (idx + 1) % 20 == 0:
            print(f"  Processed {idx+1}/{len(patches)} patches")
    
    print(f"✓ Land cover map created: {len(landcover_map)} patches classified")
    
    return landcover_map

def detect_deforestation(map_before, map_after, year_before=2018, year_after=2024):
    """
    Compare two land cover maps and identify deforestation
    Deforestation = was Forest in earlier year, became something else later
    """
    print(f"\nDetecting deforestation ({year_before} → {year_after})...")
    
    deforestation_events = []
    preserved_forest = 0
    lost_forest_pixels = 0
    
    for position in map_before.keys():
        class_before = map_before[position]['class']
        class_after = map_after.get(position, None)
        
        if class_after is None:
            continue
        
        class_after = class_after['class']
        
        # Check if forest was converted to non-forest
        if class_before == 'Forest' and class_after != 'Forest':
            deforestation_events.append({
                'position': position,
                'was': class_before,
                'became': class_after,
                'confidence_before': map_before[position]['confidence'],
                'confidence_after': map_after[position]['confidence']
            })
            lost_forest_pixels += 1
        
        # Count preserved forest
        if class_before == 'Forest' and class_after == 'Forest':
            preserved_forest += 1
    
    return deforestation_events, preserved_forest, lost_forest_pixels

def visualize_deforestation(image_path_before, image_path_after, map_2018, map_2024, year_before, year_after):
    """
    Create a visualization showing where deforestation occurred
    """
    img_before = Image.open(image_path_before)
    img_after = Image.open(image_path_after)
    
    # Create a heatmap of changes
    img_changes = img_before.copy()
    pixels = img_changes.load()
    
    # Mark deforested areas in red
    for position, data_2024 in map_2024.items():
        x, y = position
        if data_2024['class'] != 'Forest':
            if position in map_2018 and map_2018[position]['class'] == 'Forest':
                # This was forest, now it's not - mark it red
                for px in range(x, min(x + 64, 512)):
                    for py in range(y, min(y + 64, 512)):
                        pixels[px, py] = (255, 0, 0)
    
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    
    axes[0].imshow(img_before)
    axes[0].set_title(f'Satellite Image {year_before}', fontsize=12, fontweight='bold')
    axes[0].axis('off')
    
    axes[1].imshow(img_after)
    axes[1].set_title(f'Satellite Image {year_after}', fontsize=12, fontweight='bold')
    axes[1].axis('off')
    
    axes[2].imshow(img_changes)
    axes[2].set_title(f'Deforestation Areas (in red)', fontsize=12, fontweight='bold')
    axes[2].axis('off')
    
    plt.tight_layout()
    save_path = results_path / f'deforestation_visualization_{year_before}_{year_after}.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"✓ Visualization saved: {save_path}")
    plt.close()
# ============================================
# MAIN WORKFLOW
# ============================================

print("\n" + "="*60)
print("DEFORESTATION DETECTION PIPELINE")
print("="*60)

# Define image paths
image_2018_path = data_path / 'image_2018.png'
image_2024_path = data_path / 'image_2024.png'

# Check if test images exist
if not image_2018_path.exists() or not image_2024_path.exists():
    print(f"\n❌ Test images not found!")
    print(f"Please run this first to generate test data:")
    print(f"   python generate_test_images.py")
    exit()

# Create land cover maps for both years
print("\n" + "="*60)
print("STEP 1: Creating land cover maps")
print("="*60)

map_2018 = create_landcover_map(image_2018_path, "2018 baseline")
map_2024 = create_landcover_map(image_2024_path, "2024 current")

# DEBUG: Print all forest patches detected
print("\n=== DEBUG: Forest patches in 2018 ===")
for pos, data in map_2018.items():
    if data['class'] == 'Forest':
        print(f"Position {pos}: {data['class']} (confidence: {data['confidence']:.2f})")

if map_2018 is None or map_2024 is None:
    print("❌ Failed to create land cover maps")
    exit()

# Detect deforestation
print("\n" + "="*60)
print("STEP 2: Analyzing land cover changes")
print("="*60)

deforestation_events, preserved_forest, lost_forest = detect_deforestation(
    map_2018, map_2024, year_before=2018, year_after=2024
)

# Generate report
print("\n" + "="*60)
print("DEFORESTATION DETECTION REPORT (2018-2024)")
print("="*60)

total_forest_2018 = sum(1 for patch in map_2018.values() if patch['class'] == 'Forest')
total_forest_2024 = sum(1 for patch in map_2024.values() if patch['class'] == 'Forest')

forest_lost = total_forest_2018 - total_forest_2024
forest_loss_percent = (forest_lost / total_forest_2018 * 100) if total_forest_2018 > 0 else 0

print(f"\nForest Statistics:")
print(f"  2018: {total_forest_2018} patches (100%)")
if total_forest_2018 > 0:
    print(f"  2024: {total_forest_2024} patches ({100 * total_forest_2024 / total_forest_2018:.1f}%)")
else:
    print(f"  2024: {total_forest_2024} patches (no forest in 2018)")
print(f"  Lost: {forest_lost} patches ({forest_loss_percent:.1f}%)")

print(f"\nDeforestation Events: {len(deforestation_events)} patches converted")

if len(deforestation_events) > 0:
    print(f"\nConversion breakdown:")
    conversions = {}
    for event in deforestation_events:
        became = event['became']
        conversions[became] = conversions.get(became, 0) + 1
    
    for land_type, count in sorted(conversions.items(), key=lambda x: -x[1]):
        percent = 100 * count / len(deforestation_events)
        print(f"  Forest → {land_type}: {count} patches ({percent:.1f}%)")

print(f"\nPreserved Forest Patches: {preserved_forest}")

# Visualize results
print("\n" + "="*60)
print("STEP 3: Creating visualizations")
print("="*60)

visualize_deforestation(image_2018_path, image_2024_path, map_2018, map_2024, 2018, 2024)
# Save detailed report
report_path = results_path / 'deforestation_report.txt'
with open(report_path, 'w', encoding='utf-8') as f:
    f.write("DEFORESTATION DETECTION REPORT (2018-2024)\n")
    f.write("=" * 60 + "\n\n")
    f.write(f"Forest in 2018: {total_forest_2018} patches\n")
    f.write(f"Forest in 2024: {total_forest_2024} patches\n")
    f.write(f"Forest lost: {forest_lost} patches ({forest_loss_percent:.1f}%)\n")
    f.write(f"Deforestation events: {len(deforestation_events)}\n\n")
    f.write("Deforestation Zones:\n")
    for i, event in enumerate(deforestation_events, 1):
        x, y = event['position']
        f.write(f"  {i}. Position ({x}, {y}): Forest → {event['became']}\n")

print(f"✓ Report saved: {report_path}")

print("\n" + "="*60)
print("✓ DEFORESTATION DETECTION COMPLETE")
print("="*60)
print(f"\nResults saved to: {results_path}/")
print(f"  - deforestation_visualization_2018_2024.png")
print(f"  - deforestation_report.txt")