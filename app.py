"""
CIFAR-10 Image Classifier - Streamlit app.

Run with:  streamlit run app.py

Uses the saved baseline_cnn.keras and residual_cnn.keras models from the
notebook (CIFAR10_Image_Classification.ipynb). Nothing is retrained.
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from PIL import Image
from plotly.subplots import make_subplots

import cifar_utils as cu

st.set_page_config(page_title="CIFAR-10 Classifier", page_icon="🖼️", layout="wide")

LOW_CONFIDENCE = 0.50  # below this we warn the user
ACCEPTED_TYPES = ["png", "jpg", "jpeg", "webp", "bmp"]


# --------------------------------------------------------------------------- #
# Cached resources
# --------------------------------------------------------------------------- #
@st.cache_resource(show_spinner="Loading model...")
def get_model(key: str):
    return cu.load_keras_model(key)


@st.cache_resource(show_spinner="Measuring model...")
def get_facts(key: str) -> dict:
    model = get_model(key)
    facts = cu.model_facts(key, model)
    facts["latency_ms"] = cu.measure_latency_ms(model)
    return facts


@st.cache_resource(show_spinner="Loading CIFAR-10 test set...")
def get_cifar_test():
    return cu.load_cifar_test()


@st.cache_resource(show_spinner="Evaluating both models on 10,000 test images...")
def run_full_evaluation():
    x_test, y_test, source = get_cifar_test()
    results = {k: cu.evaluate_on_test_set(get_model(k), x_test, y_test) for k in cu.MODELS}
    return results, y_test, source


def load_all_models() -> None:
    """Load both models up front and fail with a clear message if one is missing."""
    try:
        for key in cu.MODELS:
            get_model(key)
    except FileNotFoundError as err:
        st.error(str(err))
        st.stop()
    except Exception as err:  # corrupted file, TF/Keras version mismatch, ...
        st.error(f"Could not load the saved models: {err}")
        st.info("The models were saved with TensorFlow 2.21 / Keras 3. "
                "Install the versions in requirements.txt.")
        st.stop()


# --------------------------------------------------------------------------- #
# Chart helpers
# --------------------------------------------------------------------------- #
def style(fig: go.Figure, height: int = 320) -> go.Figure:
    fig.update_layout(
        height=height,
        margin=dict(l=10, r=10, t=40, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        legend=dict(orientation="h", y=-0.2),
    )
    return fig


def top3_chart(preds: list[tuple[str, float]], color: str) -> go.Figure:
    names = [n for n, _ in preds][::-1]
    values = [p * 100 for _, p in preds][::-1]
    fig = go.Figure(go.Bar(
        x=values, y=names, orientation="h", marker_color=color,
        text=[f"{v:.1f}%" for v in values], textposition="outside", cliponaxis=False,
        hovertemplate="%{y}: %{x:.2f}%<extra></extra>",
    ))
    fig.update_xaxes(range=[0, 115], title="Confidence (%)", showgrid=False)
    fig.update_yaxes(showgrid=False)
    return style(fig, height=190)


# --------------------------------------------------------------------------- #
# Sidebar
# --------------------------------------------------------------------------- #
with st.sidebar:
    st.header("Settings")
    run_choice = st.radio(
        "Models to run",
        ["Both (compare)", cu.MODELS["baseline"].label, cu.MODELS["residual"].label],
    )
    resize_mode = st.radio(
        "Resizing",
        ["Center crop", "Stretch"],
        help="Images are shrunk to 32x32 pixels. 'Center crop' keeps proportions by "
             "cutting a centered square; 'Stretch' squashes the whole image.",
    )
    st.divider()
    st.caption("**The models only know these 10 classes:**")
    st.caption(", ".join(cu.CLASS_NAMES))

st.title("CIFAR-10 Image Classifier")
st.caption("Upload a photo and compare a baseline CNN with a residual CNN, "
           "both loaded from the saved `.keras` files (no retraining).")

load_all_models()

selected_keys = (
    list(cu.MODELS) if run_choice.startswith("Both")
    else [k for k, s in cu.MODELS.items() if s.label == run_choice]
)

tab_classify, tab_compare, tab_about = st.tabs(["Classify an image", "Model comparison", "About"])


# --------------------------------------------------------------------------- #
# Tab 1 - Classify
# --------------------------------------------------------------------------- #
with tab_classify:
    uploaded = st.file_uploader("Upload an image", type=ACCEPTED_TYPES)

    with st.expander("No image handy? Try a random CIFAR-10 test image"):
        st.caption("Uses the CIFAR-10 test set (a local `cifar-10-batches-py` folder if you "
                   "have one, otherwise a one-time ~163 MB download). Shows the true label too.")
        if st.button("Pick a random test image"):
            try:
                x_test, y_test, _ = get_cifar_test()
                i = int(np.random.randint(len(x_test)))
                st.session_state["sample"] = {
                    "image": Image.fromarray(x_test[i]),
                    "label": cu.CLASS_NAMES[int(y_test[i])],
                }
            except Exception as err:
                st.warning(f"Couldn't load CIFAR-10 here ({type(err).__name__}). "
                           "Upload your own image instead.")

    source_image, true_label = None, None
    if uploaded is not None:
        try:
            source_image = Image.open(uploaded)
            source_image.load()
        except Exception:
            st.error("That file couldn't be read as an image. Try a PNG or JPG.")
            source_image = None
    elif "sample" in st.session_state:
        source_image = st.session_state["sample"]["image"]
        true_label = st.session_state["sample"]["label"]

    if source_image is None:
        st.info("Upload an image to see predictions, confidence scores and the top-3 classes.")
    else:
        batch, model_view = cu.preprocess_image(
            source_image, mode="crop" if resize_mode == "Center crop" else "stretch"
        )

        img_col, res_col = st.columns([1, 3], gap="large")
        with img_col:
            st.image(cu.to_rgb(source_image), caption="Your image", width="stretch")
            st.image(model_view.resize((160, 160), Image.NEAREST),
                     caption="What the model sees (32x32)")
            if true_label:
                st.success(f"True label: **{true_label}**")

        with res_col:
            results = {}
            cols = st.columns(len(selected_keys), gap="large")
            for col, key in zip(cols, selected_keys):
                spec = cu.MODELS[key]
                probs = cu.predict_probs(get_model(key), batch)[0]
                preds = cu.top_k(probs, 3)
                results[key] = preds
                with col:
                    st.subheader(spec.label)
                    name, conf = preds[0]
                    c1, c2 = st.columns(2)
                    c1.metric("Prediction", name)
                    c2.metric("Confidence", f"{conf * 100:.1f}%")
                    if conf < LOW_CONFIDENCE:
                        st.warning("Low confidence - the model is unsure about this one.")
                    if true_label:
                        (st.success if name == true_label else st.error)(
                            "Correct" if name == true_label else f"Wrong (true: {true_label})"
                        )
                    st.caption("Top-3 predictions")
                    st.plotly_chart(top3_chart(preds, spec.color), width="stretch",
                                    key=f"top3_{key}")
                    with st.expander("All 10 class probabilities"):
                        table = pd.DataFrame({
                            "Class": cu.CLASS_NAMES,
                            "Probability (%)": (probs * 100).round(2),
                        }).sort_values("Probability (%)", ascending=False)
                        st.dataframe(table, hide_index=True, width="stretch")

            if len(results) == 2:
                b, r = results["baseline"][0], results["residual"][0]
                if b[0] == r[0]:
                    st.success(f"Both models agree: **{b[0]}** "
                               f"(baseline {b[1]*100:.1f}%, residual {r[1]*100:.1f}%).")
                else:
                    st.warning(f"The models disagree - baseline: **{b[0]}** ({b[1]*100:.1f}%), "
                               f"residual: **{r[0]}** ({r[1]*100:.1f}%).")

        st.caption("Confidence is the softmax probability, not a guarantee of correctness. "
                   "A photo of something outside the 10 classes will still be forced into one of them.")


# --------------------------------------------------------------------------- #
# Tab 2 - Comparison dashboard
# --------------------------------------------------------------------------- #
with tab_compare:
    facts = {k: get_facts(k) for k in cu.MODELS}
    base, res = cu.MODELS["baseline"], cu.MODELS["residual"]

    st.subheader("Headline results")
    st.caption("Test accuracy and loss are the values printed in the notebook "
               "(10,000 CIFAR-10 test images).")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Baseline test accuracy", f"{base.test_accuracy*100:.2f}%")
    c2.metric("Residual test accuracy", f"{res.test_accuracy*100:.2f}%",
              delta=f"{(res.test_accuracy-base.test_accuracy)*100:+.2f} pts vs baseline")
    c3.metric("Baseline test loss", f"{base.test_loss:.4f}")
    c4.metric("Residual test loss", f"{res.test_loss:.4f}",
              delta=f"{res.test_loss-base.test_loss:+.4f}", delta_color="inverse")

    summary = pd.DataFrame({
        "Metric": ["Test accuracy", "Test loss", "Best validation accuracy",
                   "Total parameters", "Trainable parameters", "Model file size (MB)",
                   "Inference time, 1 image (ms, this machine)"],
        base.label: [f"{base.test_accuracy*100:.2f}%", f"{base.test_loss:.4f}",
                     f"{max(base.val_acc)*100:.2f}%", f"{facts['baseline']['total_params']:,}",
                     f"{facts['baseline']['trainable_params']:,}",
                     f"{facts['baseline']['file_size_mb']:.2f}",
                     f"{facts['baseline']['latency_ms']:.1f}"],
        res.label: [f"{res.test_accuracy*100:.2f}%", f"{res.test_loss:.4f}",
                    f"{max(res.val_acc)*100:.2f}%", f"{facts['residual']['total_params']:,}",
                    f"{facts['residual']['trainable_params']:,}",
                    f"{facts['residual']['file_size_mb']:.2f}",
                    f"{facts['residual']['latency_ms']:.1f}"],
    })
    st.dataframe(summary, hide_index=True, width="stretch")

    st.subheader("Training curves")
    st.caption("From the notebook's training logs (10 epochs each). Solid = validation, dashed = training "
               "(training metrics are measured with dropout and augmentation on).")
    epochs = list(range(1, 11))
    fig = make_subplots(rows=1, cols=2, subplot_titles=("Accuracy", "Loss"), horizontal_spacing=0.1)
    for spec in (base, res):
        for col, (val, train) in enumerate(
                [(spec.val_acc, spec.train_acc), (spec.val_loss, spec.train_loss)], start=1):
            fig.add_scatter(x=epochs, y=val, mode="lines+markers", name=f"{spec.label} (val)",
                            line=dict(color=spec.color), legendgroup=spec.key,
                            showlegend=(col == 1), row=1, col=col)
            fig.add_scatter(x=epochs, y=train, mode="lines", name=f"{spec.label} (train)",
                            line=dict(color=spec.color, dash="dash"), legendgroup=spec.key,
                            showlegend=(col == 1), row=1, col=col)
    fig.update_xaxes(title="Epoch", dtick=1)
    st.plotly_chart(style(fig, 360), width="stretch")

    st.subheader("What the numbers say")
    gap = (base.test_accuracy - res.test_accuracy) * 100
    st.markdown(
        f"- The **baseline CNN is {gap:.1f} points more accurate** on the test set "
        f"({base.test_accuracy*100:.2f}% vs {res.test_accuracy*100:.2f}%), with a similar parameter count.\n"
        f"- The residual model's **best epoch was its last (10 of 10)** and validation accuracy was still "
        f"climbing, so it was cut off by the epoch budget rather than converged. "
        f"The result shows it is *undertrained here*, not that residual connections don't help.\n"
        f"- Its architecture is also shallower than the name suggests (one residual block) and its Dense head "
        f"has no BatchNorm, unlike the baseline's. Those differences are confounded with the shortcut connection.\n"
        f"- **Ideas to test next:** more epochs, more residual blocks, BatchNorm in the head, "
        f"and a learning-rate schedule."
    )

    st.subheader("Live evaluation on the CIFAR-10 test set (optional)")
    st.caption("Re-runs the saved models on all 10,000 test images (inference only) to get per-class accuracy, "
               "confusion matrices, and where the two models agree. Needs CIFAR-10 locally or a one-time download.")
    if st.button("Run live evaluation"):
        try:
            st.session_state["eval"] = run_full_evaluation()
        except Exception as err:
            st.warning(f"Couldn't load the CIFAR-10 test set ({type(err).__name__}: {err}). "
                       "Put the extracted `cifar-10-batches-py` folder next to app.py, or connect to the internet.")

    if "eval" in st.session_state:
        ev, y_test, source = st.session_state["eval"]
        st.caption(f"Data source: {source}.")
        e1, e2, e3, e4 = st.columns(4)
        e1.metric("Baseline accuracy", f"{ev['baseline']['accuracy']*100:.2f}%")
        e2.metric("Residual accuracy", f"{ev['residual']['accuracy']*100:.2f}%")
        e3.metric("Baseline top-3 accuracy", f"{ev['baseline']['top3_accuracy']*100:.2f}%")
        e4.metric("Residual top-3 accuracy", f"{ev['residual']['top3_accuracy']*100:.2f}%")

        st.markdown("**Per-class accuracy**")
        fig = go.Figure()
        for spec in (base, res):
            fig.add_bar(x=cu.CLASS_NAMES, y=[v * 100 for v in ev[spec.key]["per_class_accuracy"]],
                        name=spec.label, marker_color=spec.color)
        fig.update_layout(barmode="group", yaxis_title="Accuracy (%)")
        st.plotly_chart(style(fig, 340), width="stretch")

        st.markdown("**Confusion matrices** (rows = true class, share of each row)")
        hm_cols = st.columns(2)
        for col, spec in zip(hm_cols, (base, res)):
            cm = ev[spec.key]["confusion"]
            norm = cm / cm.sum(axis=1, keepdims=True)
            hm = go.Figure(go.Heatmap(
                z=norm, x=cu.CLASS_NAMES, y=cu.CLASS_NAMES, colorscale="Blues", zmin=0, zmax=1,
                hovertemplate="True %{y} -> predicted %{x}: %{z:.1%}<extra></extra>",
            ))
            hm.update_yaxes(autorange="reversed", title="True")
            hm.update_xaxes(title="Predicted")
            hm.update_layout(title=spec.label)
            col.plotly_chart(style(hm, 420), width="stretch", key=f"cm_{spec.key}")

        st.markdown("**Most common mistakes**")
        mist_cols = st.columns(2)
        for col, spec in zip(mist_cols, (base, res)):
            cm = ev[spec.key]["confusion"].copy()
            np.fill_diagonal(cm, 0)
            order = np.dstack(np.unravel_index(np.argsort(cm, axis=None)[::-1], cm.shape))[0][:5]
            table = pd.DataFrame([{
                "True": cu.CLASS_NAMES[t], "Predicted as": cu.CLASS_NAMES[p], "Count": int(cm[t, p]),
            } for t, p in order])
            col.caption(spec.label)
            col.dataframe(table, hide_index=True, width="stretch")

        correct_b = ev["baseline"]["y_pred"] == y_test
        correct_r = ev["residual"]["y_pred"] == y_test
        st.markdown("**Agreement between the two models**")
        a1, a2, a3, a4 = st.columns(4)
        a1.metric("Both correct", f"{(correct_b & correct_r).mean()*100:.1f}%")
        a2.metric("Only baseline correct", f"{(correct_b & ~correct_r).mean()*100:.1f}%")
        a3.metric("Only residual correct", f"{(~correct_b & correct_r).mean()*100:.1f}%")
        a4.metric("Both wrong", f"{(~correct_b & ~correct_r).mean()*100:.1f}%")

    with st.expander("Layer-by-layer model summaries"):
        s1, s2 = st.columns(2)
        for col, spec in zip((s1, s2), (base, res)):
            col.caption(f"{spec.label}: {spec.blurb}")
            col.code(cu.model_summary_text(get_model(spec.key)), language="text")


# --------------------------------------------------------------------------- #
# Tab 3 - About
# --------------------------------------------------------------------------- #
with tab_about:
    st.subheader("About this project")
    st.markdown(
        """
**Task:** classify 32x32 colour images into the 10 CIFAR-10 classes.

**Models** (trained in `CIFAR10_Image_Classification.ipynb`, saved as `.keras`):
- **Baseline CNN** - two Conv-BN-ReLU blocks with max-pooling and dropout, then a Dense head.
- **Residual CNN** - a convolutional stem, one residual block with an identity shortcut, then a Dense head.

**Training setup (from the notebook):** 45,000 train / 5,000 validation split, 10,000 test images, Adam (lr 1e-3),
random horizontal flip + small rotation augmentation, up to 10 epochs with checkpointing,
learning-rate reduction and early stopping.

**How an uploaded image is handled:** converted to RGB, center-cropped (or stretched) to 32x32, divided by 255, then
passed to the model, which is the same scaling used in training.

**Limitations worth knowing**
- Models were trained on tiny 32x32 images; large, cluttered or unusual photos lose a lot of detail when shrunk.
- Anything outside the 10 classes is still assigned one of them, and softmax confidence can be high even when wrong.
- Neither model is tuned for accuracy (the notebook trains for 10 epochs), so expect mistakes on cat/dog and similar pairs.
        """
    )
