# CIFAR-10 Image Classification

Classify images into the 10 CIFAR-10 classes (airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck)
with two TensorFlow/Keras CNNs, and compare them in an interactive Streamlit app.

| Model | Description | Test accuracy |
|---|---|---|
| Baseline CNN | Two Conv-BN-ReLU blocks, max-pooling, dropout, Dense head | **75.24%** |
| Residual CNN | Conv stem + one residual block (identity shortcut), Dense head | 58.96% |

Both models were trained in `CIFAR10_Image_Classification.ipynb` and saved as `.keras` files.
**The app only loads the saved models; nothing is retrained.**

## App features

- **Image upload and prediction** - PNG, JPG, JPEG, WebP or BMP. Images are converted to RGB, center-cropped (or stretched)
  to 32x32 and scaled to 0-1, exactly like the notebook's preprocessing. A preview shows what the model actually sees.
- **Confidence score and top-3 predictions** for each model, with a low-confidence warning below 50%.
- **Side-by-side mode** that runs both models on the same image and flags agreement or disagreement.
- **Model comparison dashboard** - test accuracy/loss, parameter counts, file sizes, live inference time on your machine,
  training curves from the notebook logs, and written takeaways.
- **Optional live evaluation** on the CIFAR-10 test set: per-class accuracy, top-3 accuracy, confusion matrices,
  most common mistakes, and how often the two models agree.
- **Random CIFAR-10 test image** button that also shows the true label.

## Project structure

```
.
├── app.py                              # Streamlit app (UI)
├── cifar_utils.py                      # Model loading, preprocessing, prediction, evaluation helpers
├── CIFAR10_Image_Classification.ipynb  # Original training notebook (unchanged)
├── baseline_cnn.keras                  # Saved baseline model
├── residual_cnn.keras                  # Saved residual model
├── requirements.txt
└── README.md
```

The `.keras` files can sit next to `app.py` or in a `models/` folder.

## Run locally in VS Code

Use Python 3.10-3.12 (the models were saved with TensorFlow 2.21 / Keras 3).

1. Open the project folder in VS Code (**File > Open Folder**).
2. Open a terminal (**Terminal > New Terminal**) and create a virtual environment:

   ```bash
   python -m venv .venv
   ```

   Activate it:

   - Windows (PowerShell): `.venv\Scripts\Activate.ps1`
   - Windows (cmd): `.venv\Scripts\activate.bat`
   - macOS / Linux: `source .venv/bin/activate`

3. Select the interpreter: **Ctrl/Cmd+Shift+P > Python: Select Interpreter > `.venv`**.
4. Install dependencies and start the app:

   ```bash
   pip install -r requirements.txt
   streamlit run app.py
   ```

5. Your browser opens at <http://localhost:8501>. The first load takes a few seconds while TensorFlow starts.

### Optional: CIFAR-10 data for live evaluation

The "Run live evaluation" and "random test image" features need the CIFAR-10 test set. Either:

- do nothing, and Keras downloads it once (~163 MB, cached in `~/.keras/datasets`), or
- extract `cifar-10-python.tar.gz` (the file used in the notebook) so that a `cifar-10-batches-py/` folder sits next to `app.py`.

Everything else works without the dataset.

## Results and what they show

The baseline is 16.3 points ahead of the residual model, but the comparison is not like-for-like:

- The residual model's best epoch was its **last** (10 of 10) and validation accuracy was still rising, so it was cut off
  by the epoch budget rather than converged.
- It has only one residual block and no BatchNorm in its Dense head, so the shortcut connection isn't the only difference.
- Next experiments: train longer, add residual blocks, add BatchNorm to the head, add an LR schedule.

## Limitations

- Models are trained on 32x32 images; detail in large photos is lost when shrunk.
- Anything outside the 10 classes is still forced into one of them, and softmax confidence is not a guarantee of correctness.
- Training lasted 10 epochs per model, so accuracy is below what these architectures can reach with longer training.

## Troubleshooting

| Problem | Fix |
|---|---|
| `Could not find 'baseline_cnn.keras'` | Put both `.keras` files next to `app.py` (or in `models/`). |
| Error loading the models | Install the pinned versions: `pip install -r requirements.txt` (Keras 3 / TensorFlow 2.20+). |
| `streamlit` not recognised | Activate the virtual environment, or run `python -m streamlit run app.py`. |
| Live evaluation can't load CIFAR-10 | Check your internet connection, or place `cifar-10-batches-py/` next to `app.py`. |
