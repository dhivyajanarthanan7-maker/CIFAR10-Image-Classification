# CIFAR-10 Image Classification with Streamlit

## Project Overview
An interactive image classification application using TensorFlow/Keras CNN models trained on the CIFAR-10 dataset. The Streamlit dashboard allows users to upload images, view predictions, compare models, and explore confidence scores.

## Models and Results

| Model | Test Accuracy |
|---|---:|
| Baseline CNN | 75.24% |
| Residual CNN | 58.96% |

The baseline CNN performed better in the current experiment. The residual model requires further architecture and training improvements.

## Features
- Upload PNG, JPG, JPEG, WebP, and BMP images.
- Predict image classes using the saved CNN models.
- Display confidence scores and top-3 predictions.
- Compare Baseline CNN and Residual CNN side by side.
- Preview the original image and its resized 32×32 version.
- Explore class probabilities and model performance.

## CIFAR-10 Classes
Airplane, automobile, bird, cat, deer, dog, frog, horse, ship, and truck.

## Technologies
Python, TensorFlow, Keras, NumPy, Matplotlib, Streamlit.

## Project Structure
- `app.py` — Streamlit interface
- `cifar_utils.py` — Image preprocessing, predictions, and evaluation
- `CIFAR10_Image_Classification.ipynb` — Model training notebook
- `baseline_cnn.keras` — Saved baseline model
- `residual_cnn.keras` — Saved residual model
- `requirements.txt` — Dependencies

## Run Locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

Open `http://localhost:8501` in your browser.

The application loads the saved models and does not retrain them.

## Limitations and Future Improvements
- The models accept only the ten CIFAR-10 classes.
- Large images lose detail when resized to 32×32.
- Confidence scores do not guarantee correctness.
- Future work could include deeper residual networks, hyperparameter tuning, and improved model calibration.