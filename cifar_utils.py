"""
Helper functions for the CIFAR-10 Streamlit app.

Everything the app needs that is *not* UI lives here: locating and loading the
saved `.keras` models, preprocessing uploaded images exactly the way the
notebook preprocessed CIFAR-10 (RGB, 32x32, pixels scaled to 0-1), running
predictions, and (optionally) evaluating both models on the CIFAR-10 test set.

No model is ever trained here - the saved models are loaded as-is.
"""

from __future__ import annotations

import os
import pickle
import time
from dataclasses import dataclass, field
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")  # quieter TensorFlow logs

import numpy as np
from PIL import Image, ImageOps

PROJECT_DIR = Path(__file__).resolve().parent

# Same class order as the notebook (CIFAR-10 label index -> name).
CLASS_NAMES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
]
IMG_SIZE = 32


# --------------------------------------------------------------------------- #
# Model registry
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class ModelSpec:
    key: str
    label: str
    filename: str
    color: str
    blurb: str
    # Numbers copied from the notebook's printed outputs (no retraining).
    test_accuracy: float
    test_loss: float
    # Per-epoch history parsed from the notebook's training logs.
    train_acc: list = field(default_factory=list)
    train_loss: list = field(default_factory=list)
    val_acc: list = field(default_factory=list)
    val_loss: list = field(default_factory=list)


MODELS: dict[str, ModelSpec] = {
    "baseline": ModelSpec(
        key="baseline",
        label="Baseline CNN",
        filename="baseline_cnn.keras",
        color="#2A6FDB",
        blurb="Two Conv-BN-ReLU blocks (32 -> 64 filters, two convs each) "
              "with max-pooling and dropout, then a Dense(128) head with BatchNorm.",
        test_accuracy=0.7524,
        test_loss=0.7104,
        train_acc=[0.4534, 0.5770, 0.6339, 0.6636, 0.6826, 0.6982, 0.7092, 0.7207, 0.7298, 0.7358],
        train_loss=[1.5314, 1.1824, 1.0370, 0.9550, 0.8992, 0.8592, 0.8300, 0.7953, 0.7745, 0.7554],
        val_acc=[0.3256, 0.5786, 0.6380, 0.6522, 0.6698, 0.6982, 0.7358, 0.7612, 0.7214, 0.6718],
        val_loss=[1.9635, 1.2088, 1.0295, 0.9763, 0.9676, 0.8655, 0.7628, 0.6948, 0.8090, 1.0438],
    ),
    "residual": ModelSpec(
        key="residual",
        label="Residual CNN",
        filename="residual_cnn.keras",
        color="#E8833A",
        blurb="Conv-BN-ReLU stem, one residual block (two 3x3 convs + identity "
              "shortcut), then a 64-filter conv and a plain Dense(128) head.",
        test_accuracy=0.5896,
        test_loss=1.2403,
        train_acc=[0.1948, 0.2703, 0.2964, 0.3071, 0.3202, 0.3246, 0.3388, 0.3626, 0.3753, 0.3970],
        train_loss=[2.1260, 1.8672, 1.8111, 1.7730, 1.7440, 1.7272, 1.6910, 1.6419, 1.6199, 1.5614],
        val_acc=[0.1970, 0.3948, 0.3792, 0.3980, 0.4496, 0.4584, 0.4796, 0.4588, 0.5150, 0.5866],
        val_loss=[2.1745, 1.6523, 1.7470, 1.6271, 1.5137, 1.5037, 1.4209, 1.4912, 1.4065, 1.2461],
    ),
}


def find_model_file(filename: str) -> Path | None:
    """Look for a saved model in the project root, then in ./models."""
    for folder in (PROJECT_DIR, PROJECT_DIR / "models"):
        candidate = folder / filename
        if candidate.is_file():
            return candidate
    return None


def load_keras_model(key: str):
    """Load a saved model (inference only - no compile, no training)."""
    import tensorflow as tf  # imported lazily so importing this module is cheap

    spec = MODELS[key]
    path = find_model_file(spec.filename)
    if path is None:
        raise FileNotFoundError(
            f"Could not find '{spec.filename}'. Put it next to app.py "
            f"(or in a 'models/' folder)."
        )
    return tf.keras.models.load_model(path, compile=False)


def model_facts(key: str, model) -> dict:
    """Parameter counts and file size for the comparison table."""
    trainable = int(sum(np.prod(w.shape) for w in model.trainable_weights))
    total = int(model.count_params())
    path = find_model_file(MODELS[key].filename)
    size_mb = path.stat().st_size / 1_048_576 if path else float("nan")
    return {
        "total_params": total,
        "trainable_params": trainable,
        "non_trainable_params": total - trainable,
        "file_size_mb": size_mb,
    }


def model_summary_text(model) -> str:
    lines: list[str] = []
    model.summary(print_fn=lines.append, line_length=100)
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Image preprocessing + prediction
# --------------------------------------------------------------------------- #
def to_rgb(image: Image.Image) -> Image.Image:
    """Honour EXIF rotation and flatten transparency onto white -> plain RGB."""
    image = ImageOps.exif_transpose(image)
    if image.mode in ("RGBA", "LA") or (image.mode == "P" and "transparency" in image.info):
        rgba = image.convert("RGBA")
        background = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
        return Image.alpha_composite(background, rgba).convert("RGB")
    return image.convert("RGB")


def preprocess_image(image: Image.Image, mode: str = "crop") -> tuple[np.ndarray, Image.Image]:
    """
    Convert any PIL image into a model-ready batch of shape (1, 32, 32, 3).

    Matches the notebook: RGB, 32x32, float32 scaled to [0, 1] (x / 255).
    `mode="crop"` center-crops to a square first (keeps aspect ratio);
    `mode="stretch"` squashes the whole image into 32x32.

    Returns (batch, the 32x32 PIL image the model actually sees).
    """
    image = to_rgb(image)
    if mode == "crop":
        image = ImageOps.fit(image, (IMG_SIZE, IMG_SIZE), method=Image.LANCZOS, centering=(0.5, 0.5))
    else:
        image = image.resize((IMG_SIZE, IMG_SIZE), Image.LANCZOS)
    batch = np.asarray(image, dtype=np.float32)[None, ...] / 255.0
    return batch, image


def predict_probs(model, batch: np.ndarray) -> np.ndarray:
    """Softmax probabilities, shape (N, 10)."""
    return np.asarray(model(batch, training=False))


def top_k(probs: np.ndarray, k: int = 3) -> list[tuple[str, float]]:
    """Top-k (class name, probability) pairs for a single probability vector."""
    idx = np.argsort(probs)[::-1][:k]
    return [(CLASS_NAMES[i], float(probs[i])) for i in idx]


def measure_latency_ms(model, runs: int = 30) -> float:
    """Median single-image inference time on this machine, in milliseconds."""
    x = np.random.rand(1, IMG_SIZE, IMG_SIZE, 3).astype("float32")
    for _ in range(3):  # warm-up (graph tracing etc.)
        predict_probs(model, x)
    times = []
    for _ in range(runs):
        start = time.perf_counter()
        predict_probs(model, x)
        times.append((time.perf_counter() - start) * 1000)
    return float(np.median(times))


# --------------------------------------------------------------------------- #
# Optional: evaluation on the CIFAR-10 test set (no training involved)
# --------------------------------------------------------------------------- #
def _load_batch(file_path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Same loader as the notebook, for the original 'cifar-10-batches-py' folder."""
    with open(file_path, "rb") as f:
        batch = pickle.load(f, encoding="bytes")
    images = batch[b"data"].reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
    return images, np.array(batch[b"labels"])


def load_cifar_test() -> tuple[np.ndarray, np.ndarray, str]:
    """
    Return (x_test, y_test, source). Uses a local 'cifar-10-batches-py' folder
    (the notebook's format) if present, otherwise Keras downloads CIFAR-10 once
    and caches it in ~/.keras/datasets. Raises on failure (e.g. offline).
    """
    for folder in (PROJECT_DIR / "cifar-10-batches-py", PROJECT_DIR / "data" / "cifar-10-batches-py"):
        test_file = folder / "test_batch"
        if test_file.is_file():
            x, y = _load_batch(test_file)
            return x, y, f"local folder ({folder.name})"

    import tensorflow as tf

    (_, _), (x, y) = tf.keras.datasets.cifar10.load_data()
    return x, y.reshape(-1), "Keras dataset download"


def evaluate_on_test_set(model, x_test: np.ndarray, y_test: np.ndarray, batch_size: int = 500) -> dict:
    """Accuracy, top-3 accuracy, loss, per-class accuracy and confusion matrix."""
    probs = np.concatenate([
        predict_probs(model, x_test[i:i + batch_size].astype("float32") / 255.0)
        for i in range(0, len(x_test), batch_size)
    ])
    y_pred = probs.argmax(axis=1)
    n_classes = len(CLASS_NAMES)

    confusion = np.zeros((n_classes, n_classes), dtype=int)
    np.add.at(confusion, (y_test, y_pred), 1)

    top3 = np.argsort(probs, axis=1)[:, -3:]
    return {
        "y_pred": y_pred,
        "accuracy": float((y_pred == y_test).mean()),
        "top3_accuracy": float((top3 == y_test[:, None]).any(axis=1).mean()),
        "loss": float(-np.log(np.clip(probs[np.arange(len(y_test)), y_test], 1e-12, 1)).mean()),
        "per_class_accuracy": (confusion.diagonal() / confusion.sum(axis=1)).tolist(),
        "confusion": confusion,
    }
