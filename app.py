"""
----------------------------------------------------------------------
Handwritten Digit Recognition — Interactive Deep Learning GUI
B.Tech Computer Science Deep Learning Project
Authors: Abhinav Vaidya, Gauri Joshi (Roll Nos: B1-03, B1-08)
Models: Improved CNN (Winner) | Basic CNN | ANN | SVM Baseline
----------------------------------------------------------------------

Run locally:
    streamlit run app.py
"""

# ─── Standard Library ──────────────────────────────────────────────────────────
import os
import json
import time
import warnings
from io import BytesIO

warnings.filterwarnings("ignore")

# ─── Third-Party ───────────────────────────────────────────────────────────────
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image, ImageOps, ImageFilter

# ─── Page Configuration (No Emojis) ───────────────────────────────────────────
st.set_page_config(
    page_title="Handwritten Digit Recognition | Deep Learning Project",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Load Custom CSS ───────────────────────────────────────────────────────────
_css_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "style.css")
if os.path.exists(_css_path):
    with open(_css_path, encoding="utf-8") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# ─── Optional / Heavy Libraries (Safe Imports) ─────────────────────────────────
try:
    from streamlit_drawable_canvas import st_canvas
    CANVAS_AVAILABLE = True
except ImportError:
    CANVAS_AVAILABLE = False

try:
    import tensorflow as tf
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False

try:
    import plotly.graph_objects as go
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

try:
    import joblib
    JOBLIB_AVAILABLE = True
except ImportError:
    JOBLIB_AVAILABLE = False

# ─── File Paths & Constants ────────────────────────────────────────────────────
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_KERAS_PATH = os.path.join(PROJECT_DIR, "best_model.keras")
MODEL_JOBLIB_PATH = os.path.join(PROJECT_DIR, "best_model.joblib")
METADATA_PATH = os.path.join(PROJECT_DIR, "best_model_metadata.json")
SAMPLES_NPZ_PATH = os.path.join(PROJECT_DIR, "mnist_samples.npz")

# Academic experimental results from comparative study (No Emojis)
DEFAULT_COMPARATIVE_RESULTS = [
    {
        "rank": "1",
        "model": "Improved CNN",
        "type": "Deep Learning (CNN + BN + Dropout)",
        "accuracy": 99.06,
        "f1": 99.05,
        "train_time": "397.6 s",
        "pred_time": "3.36 s",
        "color": "#6EE7B7",
    },
    {
        "rank": "2",
        "model": "Basic CNN",
        "type": "Deep Learning (Standard CNN)",
        "accuracy": 98.90,
        "f1": 98.89,
        "train_time": "340.4 s",
        "pred_time": "2.75 s",
        "color": "#93C5FD",
    },
    {
        "rank": "3",
        "model": "ANN",
        "type": "Deep Learning (Dense Fully-Connected)",
        "accuracy": 97.38,
        "f1": 97.36,
        "train_time": "34.6 s",
        "pred_time": "0.82 s",
        "color": "#FCA5A5",
    },
    {
        "rank": "4",
        "model": "SVM (RBF)",
        "type": "Traditional ML (Baseline)",
        "accuracy": 96.84,
        "f1": 96.82,
        "train_time": "11.9 s*",
        "pred_time": "37.66 s",
        "color": "#FDE68A",
    },
]


# ══════════════════════════════════════════════════════════════════════════════
#  Resource Loaders (Cached)
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_resource(show_spinner=False)
def load_model_and_metadata():
    """
    Loads the trained model (Keras / Joblib) and metadata JSON.
    Returns (model, metadata_dict, model_type).
    """
    metadata = {}
    if os.path.exists(METADATA_PATH):
        try:
            with open(METADATA_PATH, "r", encoding="utf-8") as f:
                metadata = json.load(f)
        except Exception:
            pass

    # 1. Try Keras model (Improved CNN)
    if TF_AVAILABLE and os.path.exists(MODEL_KERAS_PATH):
        try:
            # compile=False speeds up loading and prevents custom optimizer issues
            model = tf.keras.models.load_model(MODEL_KERAS_PATH, compile=False)
            return model, metadata, "keras"
        except Exception as e:
            st.warning(f"Note: Keras load notice: {e}")

    # 2. Try Joblib model (SVM fallback)
    if JOBLIB_AVAILABLE and os.path.exists(MODEL_JOBLIB_PATH):
        try:
            model = joblib.load(MODEL_JOBLIB_PATH)
            return model, metadata, "svm"
        except Exception:
            pass

    return None, metadata, None


@st.cache_data(show_spinner=False)
def load_mnist_test_samples():
    """
    Loads bundled test MNIST samples for quick live demonstration.
    """
    if os.path.exists(SAMPLES_NPZ_PATH):
        try:
            data = np.load(SAMPLES_NPZ_PATH)
            return data["images"], data["labels"]
        except Exception:
            pass
    return None, None


# ══════════════════════════════════════════════════════════════════════════════
#  MNIST-Standard Preprocessing Engine
# ══════════════════════════════════════════════════════════════════════════════

def preprocess_digit_image(img_input):
    """
    Standardized MNIST-compatible preprocessing pipeline:
    1. Grayscale conversion.
    2. Polarity normalization (ensures white digit on black background).
    3. Blank canvas detection.
    4. Content bounding-box extraction.
    5. Aspect-ratio preserving scaling into a 20x20 box (Lanczos anti-aliasing).
    6. Centering on 28x28 black canvas.
    7. Center-of-Mass (centroid) alignment to field center (14, 14).
    8. Pixel value normalization to [0.0, 1.0].
    
    Returns normalized numpy array of shape (28, 28) or None if blank.
    """
    if img_input is None:
        return None

    # Handle canvas RGBA numpy array or PIL Image
    if isinstance(img_input, np.ndarray):
        # Check alpha channel if available
        if img_input.ndim == 3 and img_input.shape[2] == 4:
            alpha = img_input[:, :, 3]
            if alpha.max() < 10:
                return None
        pil_img = Image.fromarray(img_input.astype(np.uint8))
    else:
        pil_img = img_input

    # 1. Convert to Grayscale
    gray_img = pil_img.convert("L")
    arr = np.array(gray_img, dtype=np.float32)

    # 2. Polarity Detection:
    # Sample corner pixels to see if background is dark or light
    h, w = arr.shape
    corner_pixels = [
        arr[0, 0], arr[0, w - 1],
        arr[h - 1, 0], arr[h - 1, w - 1]
    ]
    if np.mean(corner_pixels) > 127.0:
        # Light background -> Invert to white digit on black background
        gray_img = ImageOps.invert(gray_img)
        arr = 255.0 - arr

    # 3. Check if virtually blank
    mask = arr > 30
    if not np.any(mask) or np.count_nonzero(mask) < 8:
        return None

    # 4. Content Bounding Box Extraction
    y_indices, x_indices = np.where(mask)
    ymin, ymax = int(y_indices.min()), int(y_indices.max())
    xmin, xmax = int(x_indices.min()), int(x_indices.max())

    box_w = xmax - xmin + 1
    box_h = ymax - ymin + 1
    if box_w < 2 or box_h < 2:
        return None

    cropped = gray_img.crop((xmin, ymin, xmax + 1, ymax + 1))

    # 5. Aspect-Ratio Preserving Resize into 20x20 box (NIST/MNIST benchmark method)
    scale = 20.0 / max(box_w, box_h)
    new_w = max(1, int(round(box_w * scale)))
    new_h = max(1, int(round(box_h * scale)))
    resized_digit = cropped.resize((new_w, new_h), Image.LANCZOS)

    # 6. Paste onto 28x28 black canvas
    blank_28 = Image.new("L", (28, 28), 0)
    paste_x = (28 - new_w) // 2
    paste_y = (28 - new_h) // 2
    blank_28.paste(resized_digit, (paste_x, paste_y))

    # 7. Center of Mass (Centroid) Alignment
    digit_arr = np.array(blank_28, dtype=np.float32)
    total_mass = digit_arr.sum()
    if total_mass > 0:
        y_coords, x_coords = np.indices((28, 28))
        cy = (digit_arr * y_coords).sum() / total_mass
        cx = (digit_arr * x_coords).sum() / total_mass

        # Shift to center (14, 14), clamp shift to avoid pushing out of bounds
        shift_y = int(np.clip(np.round(14.0 - cy), -3, 3))
        shift_x = int(np.clip(np.round(14.0 - cx), -3, 3))

        if shift_y != 0:
            digit_arr = np.roll(digit_arr, shift_y, axis=0)
            if shift_y > 0:
                digit_arr[:shift_y, :] = 0
            else:
                digit_arr[shift_y:, :] = 0

        if shift_x != 0:
            digit_arr = np.roll(digit_arr, shift_x, axis=1)
            if shift_x > 0:
                digit_arr[:, :shift_x] = 0
            else:
                digit_arr[:, shift_x:] = 0

    # 8. Normalize to [0.0, 1.0]
    normalized = np.clip(digit_arr / 255.0, 0.0, 1.0).astype(np.float32)
    return normalized


# ══════════════════════════════════════════════════════════════════════════════
#  Prediction Execution
# ══════════════════════════════════════════════════════════════════════════════

def run_prediction(model, model_type, processed_28):
    """
    Executes model inference and returns (predicted_digit, probabilities_array).
    """
    if model is None:
        return mock_prediction(processed_28)

    if model_type == "keras":
        input_shape = model.input_shape
        if len(input_shape) == 4:
            x = processed_28.reshape(1, 28, 28, 1)
        else:
            x = processed_28.reshape(1, 784)

        probs = model.predict(x, verbose=0)[0]
        predicted = int(np.argmax(probs))
        return predicted, probs

    elif model_type == "svm":
        x = processed_28.reshape(1, 784)
        predicted = int(model.predict(x)[0])
        if hasattr(model, "decision_function"):
            scores = model.decision_function(x)[0]
            exp_scores = np.exp(scores - np.max(scores))
            probs = exp_scores / exp_scores.sum()
        elif hasattr(model, "predict_proba"):
            probs = model.predict_proba(x)[0]
        else:
            probs = np.zeros(10)
            probs[predicted] = 1.0
        return predicted, probs

    return mock_prediction(processed_28)


def mock_prediction(processed_28):
    """Fallback demonstration if model file is not yet placed."""
    flat = processed_28.flatten()
    seed = int(flat.sum() * 1000) % 10
    rng = np.random.RandomState(seed)
    probs = rng.dirichlet(np.ones(10) * 0.4)
    probs[seed] += 0.6
    probs /= probs.sum()
    predicted = int(np.argmax(probs))
    return predicted, probs


# ══════════════════════════════════════════════════════════════════════════════
#  Interactive Plots & Visualizations
# ══════════════════════════════════════════════════════════════════════════════

def render_confidence_chart(probs, predicted):
    """Renders interactive Plotly horizontal bar chart for all 10 digits."""
    digits = [str(d) for d in range(10)]
    percentages = [float(p) * 100 for p in probs]
    colors = ["#6EE7B7" if i == predicted else "#334155" for i in range(10)]

    if PLOTLY_AVAILABLE:
        fig = go.Figure(
            go.Bar(
                x=percentages,
                y=digits,
                orientation="h",
                marker=dict(
                    color=colors,
                    line=dict(color=["#34D399" if i == predicted else "#475569" for i in range(10)], width=1.5),
                ),
                text=[f"  {p:.1f}%" for p in percentages],
                textposition="outside",
                hovertemplate="<b>Digit %{y}</b>: %{x:.2f}%<extra></extra>",
            )
        )
        fig.update_layout(
            title=dict(text="Class Probabilities (0 to 9)", font=dict(size=15, color="#E2E8F0")),
            xaxis=dict(
                title="Confidence (%)",
                range=[0, 115],
                tickfont=dict(color="#94A3B8"),
                title_font=dict(color="#94A3B8"),
                gridcolor="#1E293B",
            ),
            yaxis=dict(
                title="Digit",
                tickfont=dict(color="#E2E8F0", size=13),
                title_font=dict(color="#94A3B8"),
                autorange="reversed",
            ),
            plot_bgcolor="#0A0F1E",
            paper_bgcolor="#0A0F1E",
            font=dict(color="#E2E8F0"),
            height=340,
            margin=dict(l=20, r=30, t=40, b=20),
            showlegend=False,
        )
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
    else:
        df = pd.DataFrame({"Digit": digits, "Confidence (%)": percentages})
        st.bar_chart(df.set_index("Digit"))


# ══════════════════════════════════════════════════════════════════════════════
#  Sidebar Layout (No Emojis)
# ══════════════════════════════════════════════════════════════════════════════

def render_sidebar(metadata, active_model_name):
    with st.sidebar:
        st.markdown(
            """
            <div class="sidebar-header">
                <div class="sidebar-badge">HDR</div>
                <div>
                    <div class="sidebar-title">Digit Recognition</div>
                    <div class="sidebar-subtitle">B.Tech Deep Learning Project</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("---")

        # Student / Project Info
        st.markdown("### Project Details")
        st.markdown(
            """
            - **Students**: Abhinav Vaidya, Gauri Joshi
            - **Roll Nos**: B1-03, B1-08
            - **Domain**: Computer Vision & Deep Learning
            - **Dataset**: MNIST (60,000 Train / 10,000 Test)
            """
        )

        st.markdown("---")

        # Active Model Card
        st.markdown("### Deployed Model")
        model_acc = metadata.get("metrics", {}).get("test_accuracy", 0.9906) * 100
        model_f1 = metadata.get("metrics", {}).get("macro_f1", 0.9905) * 100
        model_sz = metadata.get("metrics", {}).get("model_size_mb", 2.58)

        st.markdown(
            f"""
            <div class="model-info-box">
                <div class="model-name">{active_model_name} [Selected]</div>
                <div class="model-stat">Test Accuracy: <b>{model_acc:.2f}%</b></div>
                <div class="model-stat">Macro F1 Score: <b>{model_f1:.2f}%</b></div>
                <div class="model-stat">Model Size: <b>{model_sz:.2f} MB</b></div>
                <div class="model-stat">Architecture: <b>Conv2D + BN + Dropout</b></div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("---")

        # Comparative Performance Table
        st.markdown("### Model Benchmark")
        comp_data = metadata.get("comparative_results", DEFAULT_COMPARATIVE_RESULTS)
        for item in comp_data:
            m_name = item["model"]
            m_acc = item["accuracy"]
            m_f1 = item.get("f1_score", item.get("f1", 0))
            m_color = item.get("color", "#93C5FD")
            selected_tag = " [Best]" if "Improved" in m_name else ""
            st.markdown(
                f"""
                <div class="perf-card">
                    <div class="perf-name"><b>{m_name}</b>{selected_tag} ({item['type']})</div>
                    <div class="perf-metrics">
                        <span class="perf-acc">Acc: <b>{m_acc:.2f}%</b></span>
                        <span class="perf-f1">F1: <b>{m_f1:.2f}%</b></span>
                    </div>
                    <div class="perf-bar-wrap">
                        <div class="perf-bar" style="width:{max(5.0, (m_acc - 95.0) * 20.0):.1f}%; background:{m_color};"></div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.markdown("---")
        st.markdown(
            """
            <div style="font-size:0.75rem; color:#64748B; text-align:center;">
                Academic Project Demonstration<br>
                Google Colab &bull; TensorFlow &bull; Streamlit
            </div>
            """,
            unsafe_allow_html=True,
        )


# ══════════════════════════════════════════════════════════════════════════════
#  Main Application Logic
# ══════════════════════════════════════════════════════════════════════════════

def main():
    # 1. Load model & metadata
    with st.spinner("Loading Deep Learning Model..."):
        model, metadata, model_type = load_model_and_metadata()

    active_model_name = metadata.get("best_model_name", "Improved CNN")
    render_sidebar(metadata, active_model_name)

    # 2. Hero Header (No Emojis)
    st.markdown(
        """
        <div class="hero">
            <div class="hero-title">Handwritten Digit Recognition</div>
            <div class="hero-subtitle">
                Comparative Deep Learning Analysis &bull; Real-Time Inference on MNIST
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # 3. Status Badge (No Emojis)
    if model is not None:
        acc_disp = metadata.get("metrics", {}).get("test_accuracy", 0.9906) * 100
        st.markdown(
            f'<div class="status-badge status-live">Status: Model Active — '
            f'<b>{active_model_name}</b> ({acc_disp:.2f}% Test Accuracy)</div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<div class="status-badge status-demo">Status: Demo Mode — '
            '<code>best_model.keras</code> loaded in simulated mode</div>',
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # 4. Main Two-Column Layout (Input on Left | Results on Right)
    col_input, col_output = st.columns([1.15, 1], gap="large")

    image_to_predict = None
    input_source = None

    # ── Left Column: Interactive Inputs (No Emojis) ───────────────────────────
    with col_input:
        st.markdown('<div class="section-label">Input Digit</div>', unsafe_allow_html=True)

        tab_draw, tab_upload, tab_samples = st.tabs([
            "Draw Digit", "Upload Image", "MNIST Sample"
        ])

        # TAB 1: DRAW CANVAS
        with tab_draw:
            st.caption("Draw any digit (0 to 9) centered on the black canvas:")
            ctrl1, ctrl2 = st.columns([1.5, 1])
            with ctrl1:
                stroke_width = st.slider("Brush Size", min_value=12, max_value=34, value=22, step=2, key="canvas_stroke")
            with ctrl2:
                stroke_color = st.color_picker("Stroke Color", value="#FFFFFF", key="canvas_color")

            if not CANVAS_AVAILABLE:
                st.error("streamlit-drawable-canvas is not installed. Run: pip install streamlit-drawable-canvas")
                canvas_result = None
            else:
                canvas_kwargs = {
                    "fill_color": "rgba(0,0,0,0)",
                    "stroke_width": stroke_width,
                    "stroke_color": stroke_color,
                    "background_color": "#000000",
                    "height": 300,
                    "width": 300,
                    "drawing_mode": "freedraw",
                    "key": "digit_canvas",
                    "update_streamlit": True,
                }
                # Streamlit-drawable-canvas >= 0.12 requires return_image_data=True
                # to populate canvas_result.image_data without throwing RuntimeError
                try:
                    canvas_result = st_canvas(**canvas_kwargs, return_image_data=True)
                except TypeError:
                    canvas_result = st_canvas(**canvas_kwargs)

            predict_draw_btn = st.button("Predict Drawn Digit", type="primary", use_container_width=True, key="btn_draw")
            if predict_draw_btn:
                if canvas_result is not None:
                    try:
                        raw_data = canvas_result.image_data
                    except Exception:
                        raw_data = None

                    if raw_data is not None:
                        image_to_predict = raw_data
                        input_source = "Drawn Canvas"

        # TAB 2: UPLOAD IMAGE
        with tab_upload:
            st.caption("Upload a photo or scan of a single digit (PNG, JPG, JPEG):")
            uploaded_file = st.file_uploader(
                "Choose an image file",
                type=["png", "jpg", "jpeg"],
                help="Works with both dark and white backgrounds automatically.",
                key="file_uploader",
            )
            if uploaded_file is not None:
                try:
                    uploaded_pil = Image.open(uploaded_file)
                    st.image(uploaded_pil, caption="Uploaded Image", width=180)
                    predict_upload_btn = st.button("Predict Uploaded Image", type="primary", use_container_width=True, key="btn_upload")
                    if predict_upload_btn:
                        image_to_predict = uploaded_pil
                        input_source = "Uploaded Image"
                except Exception as e:
                    st.error(f"Error opening image: {e}")

        # TAB 3: MNIST TEST SAMPLES
        with tab_samples:
            st.caption("Pick a ground-truth digit from the MNIST test set for viva verification:")
            test_x, test_y = load_mnist_test_samples()
            if test_x is not None:
                selected_digit = st.selectbox(
                    "Choose Target Digit:",
                    options=list(range(10)),
                    index=3,
                    key="sample_digit_select",
                )
                sample_idx = st.slider("Sample Index", min_value=0, max_value=4, value=0, key="sample_idx_select")
                global_idx = selected_digit * 5 + sample_idx
                sample_img_arr = test_x[global_idx]
                sample_lbl = test_y[global_idx]

                st.image(Image.fromarray(sample_img_arr, mode="L").resize((140, 140), Image.NEAREST), caption=f"Ground Truth Label: {sample_lbl}", width=140)

                predict_sample_btn = st.button("Predict MNIST Sample", type="primary", use_container_width=True, key="btn_sample")
                if predict_sample_btn:
                    image_to_predict = Image.fromarray(sample_img_arr, mode="L")
                    input_source = f"MNIST Test Sample (Label: {sample_lbl})"
            else:
                st.info("Sample file mnist_samples.npz not found. Use drawing or upload mode.")

    # ── Handle Prediction Trigger ─────────────────────────────────────────────
    if image_to_predict is not None:
        processed_28 = preprocess_digit_image(image_to_predict)

        if processed_28 is None:
            st.session_state["prediction_error"] = "Please draw or select a valid digit first (canvas was blank)."
            st.session_state["prediction_done"] = False
        else:
            with st.spinner("Analyzing digit features..."):
                t0 = time.perf_counter()
                predicted, probs = run_prediction(model, model_type, processed_28)
                elapsed_ms = (time.perf_counter() - t0) * 1000.0

            confidence = float(probs[predicted]) * 100.0

            st.session_state["prediction_done"] = True
            st.session_state["prediction_error"] = None
            st.session_state["predicted"] = predicted
            st.session_state["probs"] = probs
            st.session_state["confidence"] = confidence
            st.session_state["elapsed_ms"] = elapsed_ms
            st.session_state["input_source"] = input_source
            st.session_state["processed_28"] = processed_28

    # ── Right Column: Results & Analytics (No Emojis) ─────────────────────────
    with col_output:
        st.markdown('<div class="section-label">Prediction Results</div>', unsafe_allow_html=True)

        if st.session_state.get("prediction_error"):
            st.markdown(
                f"""
                <div class="result-error">
                    <div style="font-size:1.1rem; font-weight:700; color:#FCA5A5;">[Notice] Canvas Blank</div>
                    <div style="font-size:0.85rem; color:#94A3B8;">{st.session_state['prediction_error']}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        elif st.session_state.get("prediction_done"):
            predicted = st.session_state["predicted"]
            probs = st.session_state["probs"]
            confidence = st.session_state["confidence"]
            elapsed_ms = st.session_state["elapsed_ms"]
            src_label = st.session_state["input_source"]
            processed_28 = st.session_state["processed_28"]

            # Color coding for confidence
            conf_color = "#6EE7B7" if confidence >= 80 else ("#FDE68A" if confidence >= 50 else "#FCA5A5")

            # Main Result Card
            st.markdown(
                f"""
                <div class="result-card">
                    <div class="result-label">Predicted Digit</div>
                    <div class="result-digit" style="color:{conf_color};">{predicted}</div>
                    <div class="result-confidence" style="color:{conf_color};">
                        {confidence:.2f}% Confidence
                    </div>
                    <div class="result-meta">
                        Latency: <b>{elapsed_ms:.1f} ms</b> &nbsp;|&nbsp; Model: <b>{active_model_name}</b> &nbsp;|&nbsp; Source: {src_label}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            # Interactive Plotly Chart
            render_confidence_chart(probs, predicted)

            # Two sub-columns: Probability table & Model's 28x28 View
            sub_col1, sub_col2 = st.columns([1.2, 0.8], gap="medium")

            with sub_col1:
                st.markdown("**Top Ranked Classes**")
                sorted_idx = np.argsort(probs)[::-1]
                table_rows = ""
                for rank_pos, idx in enumerate(sorted_idx[:5]):
                    pct = probs[idx] * 100.0
                    highlight = 'style="background:rgba(110,231,183,0.12); font-weight:700;"' if idx == predicted else ""
                    tag = "[Top] " if idx == predicted else ""
                    table_rows += (
                        f"<tr {highlight}>"
                        f"<td>{tag}Digit <b>{idx}</b></td>"
                        f"<td>"
                        f'  <div class="prob-bar-outer">'
                        f'    <div class="prob-bar-inner" style="width:{pct:.1f}%;"></div>'
                        f"  </div>"
                        f"</td>"
                        f'<td style="text-align:right;"><b>{pct:.2f}%</b></td>'
                        f"</tr>"
                    )

                st.markdown(
                    f"""
                    <table class="prob-table">
                        <thead>
                            <tr><th>Class</th><th>Confidence</th><th style="text-align:right;">Score</th></tr>
                        </thead>
                        <tbody>{table_rows}</tbody>
                    </table>
                    """,
                    unsafe_allow_html=True,
                )

            with sub_col2:
                st.markdown("**Model Input (28x28)**")
                # Show exactly what the CNN received after bounding box, 20x20 scaling, and center of mass shift
                preview_pil = Image.fromarray((processed_28 * 255.0).astype(np.uint8), mode="L")
                preview_large = preview_pil.resize((140, 140), Image.NEAREST)
                st.image(preview_large, caption="Centered 28x28 Grayscale", width=140)

        else:
            # Default Waiting State (No Emojis)
            st.markdown(
                """
                <div class="result-waiting">
                    <div class="result-waiting-icon">[Ready]</div>
                    <div class="result-waiting-text">
                        Draw a digit, upload an image, or select an MNIST sample,<br>
                        then click <b>Predict</b> to evaluate the neural network.
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )



if __name__ == "__main__":
    main()
