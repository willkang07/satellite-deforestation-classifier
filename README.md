# Satellite Image Classifier & Deforestation Detection

A machine learning pipeline that classifies satellite imagery into 10 land-cover types using transfer learning, then applies that classifier to detect deforestation by comparing land-cover maps across time.

## Overview

This project fine-tunes a ResNet50 convolutional neural network (pretrained on ImageNet) to classify 64x64 satellite image patches from the [EuroSAT](https://github.com/phelber/EuroSAT) dataset into 10 land-cover classes: AnnualCrop, Forest, HerbaceousVegetation, Highway, Industrial, Pasture, PermanentCrop, Residential, River, and SeaLake.

The trained classifier is then used as the backbone of a change-detection pipeline: by classifying patches from two satellite images of the same region taken at different times, the system flags patches that transitioned from `Forest` to any other class as potential deforestation events.

## Results

- **Overall test accuracy: ~94%**
- Strong performance on visually distinct classes: **Forest, Industrial, SeaLake, and Residential**
- Weaker performance on **River**, which is frequently confused with **Highway** — both appear as narrow, linear features in 64x64 patches and are genuinely difficult to distinguish from RGB alone at this resolution. This is a known, explainable limitation of the dataset/resolution rather than a modeling flaw.

See `results/confusion_matrix.png` for the full per-class breakdown.

## Tech Stack

- **Python** with **PyTorch** and **torchvision**
- **ResNet50**, pretrained on ImageNet, adapted via transfer learning (frozen backbone + fine-tuned final layer, with an optional full-unfreeze fine-tuning pass)
- **scikit-learn** for evaluation metrics (classification report, confusion matrix)
- **matplotlib / seaborn** for visualization
- Developed in **VSCode on Windows**

## Project Structure

```
├── 1_train_classifier.py       # Trains ResNet50 on EuroSAT via transfer learning
├── 1b_finetune_model.py        # Optional: unfreezes all layers for further fine-tuning
├── 2_evaluate_model.py         # Evaluates on held-out test set, generates confusion matrix
├── 3_detect_deforestation.py   # Applies the trained model to detect land-cover change
├── models/                     # Trained model weights (not included in repo)
├── results/                    # Confusion matrix and evaluation outputs
└── data/                       # EuroSAT dataset (not included in repo, see below)
```

## Setup

1. Clone this repo:
   ```
   git clone https://github.com/yourusername/satellite-deforestation-classifier.git
   cd satellite-deforestation-classifier
   ```

2. Create and activate a virtual environment:
   ```
   python -m venv venv
   venv\Scripts\activate   # Windows
   ```

3. Install dependencies:
   ```
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
   pip install numpy pandas matplotlib seaborn scikit-learn pillow
   ```

4. Download the [EuroSAT RGB dataset](https://github.com/phelber/EuroSAT) and extract it to `data/EuroSAT_RGB/`, so the structure looks like:
   ```
   data/EuroSAT_RGB/
   ├── AnnualCrop/
   ├── Forest/
   └── ...
   ```

5. Run the pipeline:
   ```
   python 1_train_classifier.py
   python 2_evaluate_model.py
   python 3_detect_deforestation.py
   ```

## Deforestation Detection

The classifier is applied to patches from two images of the same area captured at different times. Patches classified as `Forest` in the earlier image but as a different class in the later image are flagged as potential deforestation. The current pipeline has been validated using synthetic test images; integrating real Sentinel-2 imagery (e.g., via Google Earth Engine) is a natural next step for real-world validation.

## Future Work

- Integrate real multi-temporal satellite imagery (Sentinel-2 via Google Earth Engine or Copernicus Hub) in place of synthetic test images
- Explore additional spectral bands beyond RGB to help resolve the River/Highway confusion
- Quantify deforestation area/rate rather than binary patch-level flags

## Acknowledgments

- [EuroSAT dataset](https://github.com/phelber/EuroSAT) (Helber et al.)
- ResNet50 architecture, pretrained weights via torchvision