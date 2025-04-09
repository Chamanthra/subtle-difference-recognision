# DBRestNet502 - Deep Learning for Image Difference Detection

## Overview

This project implements a deep learning model called DBTResNet50, which is based on ResNet50 architecture with a custom DBT (Dynamic Batch Transformation) block, for detecting subtle differences between image pairs. The model is trained to classify differences in color, shape, and texture between "before" and "after" images.

## Project Structure

The notebook `DBRestNet502.ipynb` contains the complete implementation including:

1. Data loading and preprocessing
2. Custom dataset class (`SubtleDiffDataset`)
3. Model architecture (DBTResNet50 with DBTBlock)
4. Training and validation loops
5. Performance evaluation and visualization

## Key Features

- **Custom Dataset Handling**: The `SubtleDiffDataset` class loads and processes image pairs with their corresponding annotations.
- **DBTResNet50 Architecture**: A modified ResNet50 model with:
  - Dynamic Batch Transformation (DBT) block
  - Dual-image processing (before/after comparison)
  - Dropout regularization
- **Training Pipeline**: Includes data augmentation, learning rate scheduling, and early stopping.
- **Evaluation Metrics**: Tracks loss, accuracy, precision, recall, F1 score, and ROC curves.

## Requirements

- Python 3.x
- PyTorch
- torchvision
- numpy
- pandas
- matplotlib
- seaborn
- scikit-learn
- PIL (Pillow)

## Usage

1. Mount Google Drive (if using Colab)
2. Set the correct paths for image directories and annotation files
3. Run the notebook cells sequentially to:
   - Load and preprocess data
   - Initialize the model
   - Train the model
   - Evaluate performance
   - Visualize results

## Data Preparation

The dataset should consist of:
- Image pairs ("before" and "after" versions)
- JSON annotation files specifying differences in color, shape, and texture

## Training Parameters

- Batch size: 32
- Learning rate: 0.0001
- Weight decay: 1e-5
- StepLR scheduler (step_size=5, gamma=0.1)
- Loss function: BCEWithLogitsLoss
- Optimizer: Adam

## Results

The model achieves:
- Training accuracy: ~96.90%
- Validation accuracy: ~51.69%
- Training loss: ~0.1368
- Validation loss: ~1.1132

## Visualization

The notebook includes code to plot:
- Training/validation loss and accuracy curves
- ROC curves
- Confusion matrices

