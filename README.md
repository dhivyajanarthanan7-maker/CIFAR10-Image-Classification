# CIFAR-10 Image Classification Using CNNs

## Project Overview

This project implements and compares two deep learning models for classifying images from the CIFAR-10 dataset using TensorFlow and Keras.

The baseline model is a CNN built from scratch. The second model incorporates a residual block to investigate whether a shortcut connection improves classification performance.

## Dataset

- Dataset: CIFAR-10
- Training images: 50,000
- Test images: 10,000
- Image dimensions: 32 × 32 × 3
- Number of classes: 10

Classes: airplane, automobile, bird, cat, deer, dog, frog, horse, ship, and truck.

A validation set of 5,000 images was separated from the original training data, leaving 45,000 images for training.

## Project Structure

```text
CIFAR10_Image_Classification/
├── CIFAR10_Image_Classification.ipynb
├── baseline_cnn.keras
├── residual_cnn.keras
├── requirements.txt
└── README.md
```

## Data Pipeline

- Normalized image pixel values to the range 0–1.
- Applied data augmentation to training images.
- Used batches of 128 images.
- Used shuffling and prefetching for efficient data loading.
- Kept the official test dataset separate from training and validation.

## Model Architecture

### Baseline CNN

The baseline CNN includes:
- Convolutional layers
- Batch Normalization
- ReLU activation
- Max Pooling
- Dropout
- Dense classification layers
- Softmax output with 10 classes

### Residual CNN

The second model incorporates a residual block with a shortcut connection that adds the input to the convolutional block output.

Both models were trained using the Adam optimizer and sparse categorical cross-entropy loss. Learning-rate reduction, checkpoint saving, and early stopping were used during training.

## Experimental Results

| Metric | Baseline CNN | Residual CNN |
|---|---:|---:|
| Best validation accuracy | 76.12% | 58.66% |
| Test accuracy | 75.24% | 58.96% |
| Test loss | 0.7104 | 1.2403 |

The baseline CNN achieved 75.24% test accuracy. The residual CNN achieved 58.96% test accuracy under the current architecture and training settings.

The residual block did not improve performance in this experiment. Further investigation of architecture design, model capacity, regularization, and training settings may help improve the result.

## How to Run

1. Open `CIFAR10_Image_Classification.ipynb` in Google Colab.
2. Select **Runtime → Change runtime type → T4 GPU**, if available.
3. Run the notebook cells in order.
4. The notebook downloads and preprocesses CIFAR-10, builds both models, trains them, plots learning curves, and evaluates performance.

Install dependencies in a compatible Python environment with:

```bash
pip install -r requirements.txt
```

## Reproducibility

The notebook sets random seeds for reproducible experiments. Exact results may vary with software versions, hardware, and execution settings.

## Limitations and Future Improvements

- The residual model performed worse than the baseline in this experiment.
- The architecture and training hyperparameters could be tuned further.
- Additional experiments could investigate stronger residual architectures, learning rates, batch sizes, and regularization.

## Technologies

Python, TensorFlow, Keras, NumPy, Matplotlib, Pandas, and Google Colab.
