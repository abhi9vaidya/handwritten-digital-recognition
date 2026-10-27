# 🔢 Handwritten Digit Recognition — Streamlit UI

> **B.Tech Computer Science | Deep Learning Project :)**  
> Comparative Analysis: ANN · Basic CNN · Enhanced CNN · SVM on MNIST

---

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. (Optional) Add your trained model
Copy these files from your Colab `artifacts/` folder into this project directory:
```
d:\DL\Project\
├── best_model.keras          ← from Colab artifacts/
├── best_model_metadata.json  ← from Colab artifacts/
├── app.py
├── style.css
└── requirements.txt
```
> ⚠️ Without the model files the app runs in **Demo Mode** with a mock predictor.

### 3. Run the App
```bash
streamlit run app.py
```
Then open **http://localhost:8501** in your browser.

---

## 📋 Features

| Feature | Description |
|---|---|
| 🎨 **Drawing Canvas** | 340×340 freehand drawing area with adjustable brush size and color |
| 🔮 **Predict Digit** | One-click inference with real-time timing display |
| 📊 **Confidence Chart** | Interactive Plotly bar chart showing all 10 digit probabilities |
| 📋 **Probability Table** | Sorted breakdown of confidence scores for every digit |
| 👁️ **28×28 Preview** | Shows exactly what the model receives after preprocessing |
| 📌 **Sidebar** | Project info, model performance comparison, dataset stats |
| 🌙 **Dark Theme** | Premium dark UI with gradient hero, glassmorphism cards, smooth animations |

---

## 🏗️ Architecture

```
app.py
├── load_model_and_metadata()   # @st.cache_resource — loads Keras or SVM model
├── preprocess_canvas_image()   # RGBA → grayscale → 28×28 → [0,1]
├── run_prediction()            # Keras CNN / Keras ANN / SVM / mock
├── mock_prediction()           # Demo mode fallback
├── render_sidebar()            # Project info & performance metrics
├── render_confidence_chart()   # Plotly bar chart
└── main()                      # App layout & state management
```

---

## 🧠 Model Results

| Rank | Model | Accuracy | F1 Score | Train Time |
|---|---|---|---|---|
| 🥇 1 | Enhanced CNN | **99.40%** | **99.39%** | 137 s |
| 🥈 2 | Basic CNN | 98.98% | 98.97% | 50 s |
| 🥉 3 | SVM (RBF) | 98.31% | 98.30% | 158 s |
| 4️⃣ 4 | ANN | 97.70% | 97.69% | 35 s |

---

## 🖼️ Preprocessing Pipeline

```
User draws on canvas (340×340 RGBA)
        ↓
Convert RGBA → Grayscale
        ↓
Invert colours (white digit on black — MNIST convention)
        ↓
Crop to bounding box + 20px padding
        ↓
Gaussian blur (radius=1) to smooth jagged edges
        ↓
Resize to 28×28 (Lanczos resampling)
        ↓
Normalise pixel values: divide by 255  →  range [0, 1]
        ↓
Reshape to (1, 28, 28, 1) for CNN  or  (1, 784) for ANN/SVM
        ↓
model.predict()  →  softmax probabilities [10]
```

---

## 📦 Dependencies

```
streamlit>=1.32.0
streamlit-drawable-canvas>=0.9.3
tensorflow>=2.13.0
scikit-learn>=1.3.0
joblib>=1.3.0
Pillow>=10.0.0
numpy>=1.24.0
plotly>=5.18.0
```
