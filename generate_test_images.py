"""
Generate realistic synthetic satellite test images with texture
"""

from PIL import Image, ImageDraw, ImageFilter
import random
import numpy as np
from pathlib import Path

data_path = Path('data/sentinel_images')
data_path.mkdir(parents=True, exist_ok=True)

print("Generating realistic satellite test images...\n")

def add_satellite_texture(img, texture_strength=0.3):
    """Add noise and texture to make images look more like real satellite data"""
    img_array = np.array(img)
    
    # Add Perlin-like noise
    noise = np.random.normal(0, 20, img_array.shape)
    img_array = np.clip(img_array + noise, 0, 255).astype(np.uint8)
    
    return Image.fromarray(img_array)

def create_realistic_image(filename, forest_zones=None):
    """Create a more realistic satellite image with texture"""
    width, height = 512, 512
    patch_size = 64
    grid_size = 8
    
    img = Image.new('RGB', (width, height), color=(100, 100, 100))
    pixels = img.load()
    
    # Define more realistic colors based on Sentinel-2 satellite bands
    colors = {
        'Forest': (50, 120, 60),           # Dark green (vegetation index high)
        'AnnualCrop': (200, 180, 80),      # Golden/tan (agricultural)
        'Pasture': (140, 180, 80),         # Light green (sparse vegetation)
        'Urban': (150, 150, 150),          # Gray (concrete/buildings)
        'Barren': (180, 160, 120),         # Brown (bare soil)
    }
    
    # Create a base image with gradual color transitions
    for x in range(width):
        for y in range(height):
            # Determine which patch we're in
            grid_x = x // patch_size
            grid_y = y // patch_size
            
            # Default land type
            land_type = random.choice(['Pasture', 'AnnualCrop', 'Barren'])
            
            # Add some forest patches
            if grid_x in [1, 2, 3] and grid_y in [1, 2, 3]:
                if random.random() > 0.4:
                    land_type = 'Forest'
            
            if grid_x in [5, 6] and grid_y in [4, 5]:
                if random.random() > 0.3:
                    land_type = 'Forest'
            
            # Apply forest conversion zones (for 2024)
            if forest_zones:
                for fz_x, fz_y in forest_zones:
                    if grid_x == fz_x and grid_y == fz_y:
                        land_type = 'AnnualCrop'
            
            base_color = colors.get(land_type, (128, 128, 128))
            
            # Add realistic noise/texture
            noise_r = random.randint(-30, 30)
            noise_g = random.randint(-30, 30)
            noise_b = random.randint(-30, 30)
            
            r = max(0, min(255, base_color[0] + noise_r))
            g = max(0, min(255, base_color[1] + noise_g))
            b = max(0, min(255, base_color[2] + noise_b))
            
            pixels[x, y] = (r, g, b)
    
    # Apply slight blur for more realism
    img = img.filter(ImageFilter.GaussianBlur(radius=1))
    
    img.save(filename)
    print(f"✓ Created: {filename}")

# Generate 2018 baseline image
print("Creating 2018 baseline image...")
create_realistic_image(data_path / 'image_2018.png')

# Generate 2024 image with deforestation zones
print("Creating 2024 image (with simulated deforestation)...")

# Forest zones to convert to farmland (grid coordinates)
deforestation_zones = [(1, 2), (2, 2), (5, 4), (5, 5)]

create_realistic_image(data_path / 'image_2024.png', forest_zones=deforestation_zones)

print(f"\n{'='*60}")
print("REALISTIC SATELLITE IMAGES CREATED")
print(f"{'='*60}")
print(f"✓ Saved to: {data_path}/")
print(f"\nReady to run: python 3_detect_deforestation.py")