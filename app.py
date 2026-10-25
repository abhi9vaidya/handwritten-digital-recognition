"""
╔══════════════════════════════════════════════════════════════════════╗
║   Handwritten Digit Recognition — Streamlit UI                      ║
║   B.Tech Deep Learning Project                                       ║
║   Models: ANN | Basic CNN | Enhanced CNN | SVM                       ║
╚══════════════════════════════════════════════════════════════════════╝

Run:
    streamlit run app.py
"""

# ─── Standard Library ──────────────────────────────────────────────────────────
import os
import json
import time
import warnings

warnings.filterwarnings("ignore")

# ─── Third-party ───────────────────────────────────────────────────────────────
import numpy as np
import streamlit as st
from PIL import Image, ImageOps, ImageFilter

# ─── Page Config (MUST be the very first Streamlit call) ───────────────────────
st.set_page_config(
    page_title="Digit Recognition | Deep Learning Project",
    page_icon="🔢",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Inject custom CSS ─────────────────────────────────────────────────────────
_css_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "style.css")
if os.path.exists(_css_path):
    with open(_css_path, encoding="utf-8") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# ─── Lazy Imports (heavier libs, load after page config) ───────────────────────
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
    import joblib
    JOBLIB_AVAILABLE = True
except ImportError:
    JOBLIB_AVAILABLE = False

try:
    import plotly.graph_objects as go
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

# ─── Constants ─────────────────────────────────────────────────────────────────
MODEL_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_KERAS_PATH   = os.path.join(MODEL_DIR, "best_model.keras")
MODEL_JOBLIB_PATH  = os.path.join(MODEL_DIR, "best_model.joblib")
METADATA_PATH      = os.path.join(MODEL_DIR, "best_model_metadata.json")

# Known results from training (used in sidebar even without model file)
TRAINING_RESULTS = {
    "Enhanced CNN": {"accuracy": 99.40, "f1": 99.39, "time_s": 137.34, "color": "#6EE7B7"},
    "Basic CNN":    {"accuracy": 98.98, "f1": 98.97, "time_s":  49.51, "color": "#93C5FD"},
    "SVM (RBF)":   {"accuracy": 98.31, "f1": 98.30, "time_s": 158.37, "color": "#FDE68A"},
    "ANN":          {"accuracy": 97.70, "f1": 97.69, "time_s":  34.66, "color": "#FCA5A5"},
}


# ══════════════════════════════════════════════════════════════════════════════
#  Model Loading (cached — runs only once per session)
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_resource(show_spinner=False)
def load_model_and_metadata():
    """
    Attempts to load the saved best model and its metadata JSON.
    Returns (model, metadata_dict, model_type) or (None, {}, None) on failure.
    """
    metadata = {}
    model = None
    model_type = None

    # --- Load metadata -------------------------------------------------------
    if os.path.exists(METADATA_PATH):
        with open(METADATA_PATH, "r") as f:
            metadata = json.load(f)

    # --- Try Keras model -----------------------------------------------------
    if TF_AVAILABLE and os.path.exists(MODEL_KERAS_PATH):
        try:
            model = tf.keras.models.load_model(MODEL_KERAS_PATH)
            model_type = "keras"
            return model, metadata, model_type
        except Exception as e:
            pass  # fall through to next option

    # --- Try joblib / SVM model ----------------------------------------------
    if JOBLIB_AVAILABLE and os.path.exists(MODEL_JOBLIB_PATH):
        try:
            model = joblib.load(MODEL_JOBLIB_PATH)
            model_type = "svm"
            return model, metadata, model_type
        except Exception:
            pass

    # --- No model — demo / mock mode -----------------------------------------
    return None, metadata, None


# ══════════════════════════════════════════════════════════════════════════════
#  Image Pre-processing
# ══════════════════════════════════════════════════════════════════════════════

def preprocess_canvas_image(image_data):
    """
    Convert the raw RGBA canvas array to a normalised 28x28 grayscale numpy array.

    Steps
    -----
    1. Convert RGBA -> grayscale PIL image
    2. Invert so digit is white on black (MNIST convention)
    3. Crop to bounding box of drawn content + small padding
    4. Resize to 28x28 with Lanczos resampling
    5. Normalise pixels to [0, 1]
    6. Return shape (28, 28)
    """
    if image_data is None:
        return None

    # Grab the alpha channel as a mask to check if anything was drawn
    alpha_channel = image_data[:, :, 3]
    if alpha_channel.max() < 10:          # virtually blank canvas
        return None

    # Step 1: RGBA -> RGB -> grayscale
    pil_img = Image.fromarray(image_data.astype(np.uint8), mode="RGBA")
    pil_img = pil_img.convert("L")        # grayscale (0=black, 255=white)

    # Step 2: Invert (canvas is white bg, drawn strokes are dark)
    #         MNIST is dark bg, white digit
    pil_img = ImageOps.invert(pil_img)

    # Step 3: Crop to content bounding box
    bbox = pil_img.getbbox()
    if bbox is None:
        return None
    pad = 20
    left  = max(0, bbox[0] - pad)
    upper = max(0, bbox[1] - pad)
    right  = min(pil_img.width,  bbox[2] + pad)
    lower  = min(pil_img.height, bbox[3] + pad)
    pil_img = pil_img.crop((left, upper, right, lower))

    # Step 4: Smooth slightly before downscaling
    pil_img = pil_img.filter(ImageFilter.GaussianBlur(radius=1))

    # Step 5: Resize to 28x28
    pil_img = pil_img.resize((28, 28), Image.LANCZOS)

    # Step 6: Normalise to [0, 1]
    arr = np.array(pil_img, dtype=np.float32) / 255.0

    return arr  # shape (28, 28)


# ══════════════════════════════════════════════════════════════════════════════
#  Prediction
# ══════════════════════════════════════════════════════════════════════════════

def run_prediction(model, model_type, processed):
    """
    Run inference and return (predicted_digit, probabilities_array[10]).

    Works for:
        * Keras CNN  -- input shape (1, 28, 28, 1)
        * Keras ANN  -- input shape (1, 784)
        * SVM        -- input shape (1, 784), returns decision_function scores
    """
    if model is None:
        return mock_prediction(processed)

    if model_type == "keras":
        # Detect if the model expects flat or 2-D input
        first_layer_shape = model.input_shape
        if len(first_layer_shape) == 4:            # (None, 28, 28, 1) -> CNN
            x = processed.reshape(1, 28, 28, 1)
        else:                                       # (None, 784)       -> ANN
            x = processed.reshape(1, 784)

        probs = model.predict(x, verbose=0)[0]     # shape (10,)
        predicted = int(np.argmax(probs))
        return predicted, probs

    elif model_type == "svm":
        x = processed.reshape(1, 784)
        predicted = int(model.predict(x)[0])

        # SVM doesn't directly output probabilities.
        # Use decision_function scores, softmax-normalise for display.
        if hasattr(model, "decision_function"):
            scores = model.decision_function(x)[0]
            e = np.exp(scores - scores.max())
            probs = e / e.sum()
        elif hasattr(model, "predict_proba"):
            probs = model.predict_proba(x)[0]
        else:
            probs = np.zeros(10)
            probs[predicted] = 1.0

        return predicted, probs

    return mock_prediction(processed)


def mock_prediction(processed):
    """
    Deterministic mock using pixel sum heuristic.
    Used when no model file is found -- still gives a plausible demo.
    """
    flat = processed.flatten()
    seed = int(flat.sum() * 1000) % 10
    np.random.seed(seed)
    probs = np.random.dirichlet(np.ones(10) * 0.3)
    probs[seed] += 0.5
    probs /= probs.sum()
    predicted = int(np.argmax(probs))
    return predicted, probs


# ══════════════════════════════════════════════════════════════════════════════
#  Sidebar
# ══════════════════════════════════════════════════════════════════════════════

def render_sidebar(metadata):
    with st.sidebar:
        st.markdown(
            """
            <div class="sidebar-header">
                <span class="sidebar-icon">🔢</span>
                <div>
                    <div class="sidebar-title">Digit Recognition</div>
                    <div class="sidebar-subtitle">Deep Learning Project</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("---")

        # About
        st.markdown("### 📘 About the Project")
        st.markdown(
            """
            A **B.Tech Computer Science** deep learning project that builds and
            compares **four different models** for recognising handwritten digits
            (0 – 9) on the MNIST dataset.

            | | Model | Type |
            |---|---|---|
            | 🥇 | Enhanced CNN | Deep Learning |
            | 🥈 | Basic CNN | Deep Learning |
            | 🥉 | SVM (RBF) | Traditional ML |
            | 4️⃣ | ANN | Deep Learning |
            """
        )

        st.markdown("---")

        # Performance Table
        st.markdown("### 📊 Model Performance")
        for name, res in TRAINING_RESULTS.items():
            crown = "👑 " if name == "Enhanced CNN" else ""
            st.markdown(
                f"""
                <div class="perf-card">
                    <div class="perf-name">{crown}{name}</div>
                    <div class="perf-metrics">
                        <span class="perf-acc">Acc: {res['accuracy']:.2f}%</span>
                        <span class="perf-f1">F1: {res['f1']:.2f}%</span>
                    </div>
                    <div class="perf-bar-wrap">
                        <div class="perf-bar" style="width:{res['accuracy'] - 97:.1f}%;
                             background:{res['color']};"></div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.markdown("---")

        # Dataset Info
        st.markdown("### 🗃️ Dataset: MNIST")
        col1, col2 = st.columns(2)
        col1.metric("Train", "60,000")
        col2.metric("Test",  "10,000")

        st.markdown("---")

        # Active Model info (from metadata if present)
        st.markdown("### 🤖 Active Model")
        if metadata:
            mn = metadata.get("best_model_name", "Unknown")
            mt = metadata.get("metrics", {})
            acc = mt.get("test_accuracy", 0) * 100
            f1  = mt.get("macro_f1", 0) * 100
            sz  = mt.get("model_size_mb", 0)
            st.markdown(
                f"""
                <div class="model-info-box">
                    <div class="model-name">{mn}</div>
                    <div class="model-stat">🎯 Accuracy: <b>{acc:.2f}%</b></div>
                    <div class="model-stat">📈 F1 Score: <b>{f1:.2f}%</b></div>
                    <div class="model-stat">💾 Size: <b>{sz:.2f} MB</b></div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.info(
                "Place `best_model.keras` + `best_model_metadata.json` "
                "in the project folder to activate the real model."
            )

        st.markdown("---")
        st.markdown(
            "<div style='text-align:center;font-size:12px;color:#666;'>Built with ❤️ using Streamlit</div>",
            unsafe_allow_html=True,
        )


# ══════════════════════════════════════════════════════════════════════════════
#  Confidence Bar Chart  (Plotly)
# ══════════════════════════════════════════════════════════════════════════════

def render_confidence_chart(probs, predicted):
    """Render an animated Plotly bar chart for all 10 digit probabilities."""
    digits = list(range(10))
    colors = ["#6EE7B7" if i == predicted else "#334155" for i in digits]

    if PLOTLY_AVAILABLE:
        fig = go.Figure(
            go.Bar(
                x=[str(d) for d in digits],
                y=[float(p) * 100 for p in probs],
                marker_color=colors,
                text=[f"{p*100:.1f}%" for p in probs],
                textposition="outside",
                hovertemplate="Digit %{x}: %{y:.2f}%<extra></extra>",
            )
        )
        fig.update_layout(
            title=dict(text="Confidence Scores (%)", font=dict(size=16, color="#E2E8F0")),
            xaxis=dict(
                title="Digit",
                tickfont=dict(color="#94A3B8", size=14),
                title_font=dict(color="#94A3B8"),
            ),
            yaxis=dict(
                title="Confidence (%)",
                range=[0, 115],
                tickfont=dict(color="#94A3B8"),
                title_font=dict(color="#94A3B8"),
                gridcolor="#1E293B",
            ),
            plot_bgcolor="#0F172A",
            paper_bgcolor="#0F172A",
            font=dict(color="#E2E8F0"),
            height=340,
            margin=dict(l=20, r=20, t=50, b=20),
            showlegend=False,
        )
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
    else:
        # Fallback: Streamlit native bar chart
        import pandas as pd
        df = {"Digit": [str(d) for d in digits], "Confidence (%)": [p * 100 for p in probs]}
        st.bar_chart(df)


# ══════════════════════════════════════════════════════════════════════════════
#  Main App
# ══════════════════════════════════════════════════════════════════════════════

def main():
    # Load model once
    with st.spinner("Loading model…"):
        model, metadata, model_type = load_model_and_metadata()

    # Sidebar
    render_sidebar(metadata)

    # Hero header
    st.markdown(
        """
        <div class="hero">
            <div class="hero-title">Handwritten Digit Recognition</div>
            <div class="hero-subtitle">
                Draw a digit below &bull; Click <b>Predict Digit</b> &bull; See the magic ✨
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Model status badge
    if model is not None:
        acc_disp = metadata.get("metrics", {}).get("test_accuracy", 0) * 100
        st.markdown(
            f'<div class="status-badge status-live">✅ Model Loaded — '
            f'{metadata.get("best_model_name","Unknown")} ({acc_disp:.2f}% accuracy)</div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<div class="status-badge status-demo">⚠️ Demo Mode — place '
            '<code>best_model.keras</code> in the project folder to enable real predictions</div>',
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # ══════════════════════════════════════════════════════════════════════════
    #  Two-column layout: Canvas  |  Results
    # ══════════════════════════════════════════════════════════════════════════
    col_canvas, col_results = st.columns([1.1, 1], gap="large")

    # ── Left: Drawing Canvas ──────────────────────────────────────────────────
    with col_canvas:
        st.markdown('<div class="section-label">✏️ Draw Your Digit</div>', unsafe_allow_html=True)

        # Controls row
        ctrl_col1, ctrl_col2, _ = st.columns([1, 1, 1])
        with ctrl_col1:
            stroke_width = st.slider(
                "Brush Size", min_value=8, max_value=40, value=18, step=2,
                key="brush_slider",
            )
        with ctrl_col2:
            stroke_color = st.color_picker("Stroke Color", value="#FFFFFF", key="stroke_color")

        # Canvas
        if not CANVAS_AVAILABLE:
            st.error(
                "📦 `streamlit-drawable-canvas` is not installed.\n\n"
                "Run: `pip install streamlit-drawable-canvas`"
            )
            canvas_result = None
        else:
            canvas_result = st_canvas(
                fill_color="rgba(0,0,0,0)",
                stroke_width=stroke_width,
                stroke_color=stroke_color,
                background_color="#000000",
                height=340,
                width=340,
                drawing_mode="freedraw",
                key="digit_canvas",
                display_toolbar=False,
                update_streamlit=True,
            )

        st.markdown(
            '<div class="canvas-hint">Draw a single digit (0–9) inside the black area above</div>',
            unsafe_allow_html=True,
        )

        # Predict Button
        predict_btn = st.button(
            "🔮  Predict Digit",
            use_container_width=True,
            type="primary",
            key="predict_button",
        )

    # ── Right: Results Panel ──────────────────────────────────────────────────
    with col_results:
        st.markdown('<div class="section-label">📊 Prediction Results</div>', unsafe_allow_html=True)

        result_placeholder   = st.empty()
        chart_placeholder    = st.empty()
        table_placeholder    = st.empty()
        preview_placeholder  = st.empty()

        # Default "waiting" state
        if "prediction_done" not in st.session_state:
            result_placeholder.markdown(
                """
                <div class="result-waiting">
                    <div class="result-waiting-icon">🖊️</div>
                    <div class="result-waiting-text">Draw a digit and<br>click <b>Predict Digit</b></div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    # ══════════════════════════════════════════════════════════════════════════
    #  Handle Prediction
    # ══════════════════════════════════════════════════════════════════════════
    if predict_btn:
        image_data = (
            canvas_result.image_data
            if (canvas_result is not None and hasattr(canvas_result, "image_data"))
            else None
        )
        processed = preprocess_canvas_image(image_data)

        if processed is None:
            with col_results:
                result_placeholder.markdown(
                    """
                    <div class="result-error">
                        <div style="font-size:2.5rem;">🖊️</div>
                        <div>Please draw a digit first!</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
        else:
            with st.spinner("Analysing…"):
                t0 = time.perf_counter()
                predicted, probs = run_prediction(model, model_type, processed)
                elapsed_ms = (time.perf_counter() - t0) * 1000

            confidence = float(probs[predicted]) * 100
            mode_label = "Demo Mode" if model is None else metadata.get("best_model_name", "Model")

            # Persist to session state so results survive reruns
            st.session_state["prediction_done"] = True
            st.session_state["predicted"]   = predicted
            st.session_state["probs"]        = probs
            st.session_state["confidence"]   = confidence
            st.session_state["elapsed_ms"]   = elapsed_ms
            st.session_state["mode_label"]   = mode_label
            st.session_state["processed"]    = processed

    # Render results (from session state)
    if st.session_state.get("prediction_done"):
        predicted   = st.session_state["predicted"]
        probs       = st.session_state["probs"]
        confidence  = st.session_state["confidence"]
        elapsed_ms  = st.session_state["elapsed_ms"]
        mode_label  = st.session_state["mode_label"]
        processed   = st.session_state["processed"]

        with col_results:
            # Big digit display
            conf_color = (
                "#6EE7B7" if confidence >= 80
                else "#FDE68A" if confidence >= 50
                else "#FCA5A5"
            )
            result_placeholder.markdown(
                f"""
                <div class="result-card">
                    <div class="result-label">Predicted Digit</div>
                    <div class="result-digit" style="color:{conf_color};">{predicted}</div>
                    <div class="result-confidence" style="color:{conf_color};">
                        {confidence:.1f}% confidence
                    </div>
                    <div class="result-meta">
                        ⏱ {elapsed_ms:.1f} ms &nbsp;|&nbsp; 🤖 {mode_label}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            # Confidence bar chart
            with chart_placeholder.container():
                render_confidence_chart(probs, predicted)

            # Probability table
            with table_placeholder.container():
                st.markdown("**Probability Breakdown**")
                rows = ""
                sorted_indices = np.argsort(probs)[::-1]
                for idx in sorted_indices:
                    pct = probs[idx] * 100
                    bar_w = pct
                    highlight = 'style="background:rgba(110,231,183,0.12);font-weight:700;"' if idx == predicted else ""
                    rows += (
                        f'<tr {highlight}>'
                        f'<td>{"👑 " if idx == predicted else ""}{idx}</td>'
                        f'<td>'
                        f'  <div class="prob-bar-outer">'
                        f'    <div class="prob-bar-inner" style="width:{bar_w:.1f}%;"></div>'
                        f'  </div>'
                        f'</td>'
                        f'<td style="text-align:right;">{pct:.2f}%</td>'
                        f'</tr>'
                    )
                st.markdown(
                    f"""
                    <table class="prob-table">
                        <thead>
                            <tr><th>Digit</th><th>Confidence</th><th>Score</th></tr>
                        </thead>
                        <tbody>{rows}</tbody>
                    </table>
                    """,
                    unsafe_allow_html=True,
                )

            # 28x28 preview
            with preview_placeholder.container():
                st.markdown("**What the model sees (28×28)**")
                preview_img = Image.fromarray((processed * 255).astype(np.uint8), mode="L")
                preview_img = preview_img.resize((140, 140), Image.NEAREST)
                st.image(preview_img, caption="28×28 grayscale input", width=140)

    # ══════════════════════════════════════════════════════════════════════════
    #  Instructions Section
    # ══════════════════════════════════════════════════════════════════════════
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("---")

    st.markdown(
        """
        <div class="how-to-section">
            <div class="how-to-title">How It Works</div>
            <div class="how-to-grid">
                <div class="how-to-card">
                    <div class="how-to-num">1</div>
                    <div class="how-to-text"><b>Draw</b> any digit 0–9 on the black canvas</div>
                </div>
                <div class="how-to-card">
                    <div class="how-to-num">2</div>
                    <div class="how-to-text"><b>Click</b> the <em>Predict Digit</em> button</div>
                </div>
                <div class="how-to-card">
                    <div class="how-to-num">3</div>
                    <div class="how-to-text">
                        Image is <b>resized to 28×28</b>, converted to grayscale &amp; normalised
                    </div>
                </div>
                <div class="how-to-card">
                    <div class="how-to-num">4</div>
                    <div class="how-to-text">
                        Fed into the <b>Enhanced CNN</b> (99.40% accuracy) for prediction
                    </div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    main()
