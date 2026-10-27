# Handwritten Digit Recognition — Interactive Deep Learning GUI

> **B.Tech Computer Science | Deep Learning Project**  
> **Authors**: Abhinav Vaidya, Gauri Joshi (Roll Nos: B1-03, B1-08)  
> **Comparative Analysis**: Improved CNN (Winner) · Basic CNN · ANN · SVM Baseline on MNIST

---

## Overview

This project provides an interactive, production-grade graphical user interface (GUI) built with **Streamlit** to demonstrate handwritten digit recognition. The underlying model is the **Improved CNN**, which achieved the highest test accuracy (**99.06%**) in our comparative evaluation of deep learning architectures on the MNIST benchmark.

The application allows users to:
1. **Draw digits freehand** on an interactive black canvas.
2. **Upload custom images** (photos or scans of handwritten digits in PNG/JPG/JPEG).
3. **Test real MNIST benchmark samples** directly from the test dataset to verify ground-truth classification.
4. **Inspect complete class probability distributions (0–9)** via interactive Plotly charts and sorted confidence tables.
5. **Visualize the preprocessed 28×28 input** that the neural network actually sees after aspect-ratio scaling and center-of-mass alignment.

---

## Comparative Evaluation Results (From Academic Experiments)

| Rank | Model Architecture | Paradigm | Test Accuracy | Macro F1 | Training Time | Prediction Latency |
|:---:|---|---|:---:|:---:|:---:|:---:|
| **1** | **Improved CNN** (Conv + BN + Dropout) | Deep Learning | **99.06%** | **99.05%** | 397.6 s | **3.36 s** (10k images) |
| **2** | **Basic CNN** (Standard Conv + MaxPool) | Deep Learning | 98.90% | 98.89% | 340.4 s | 2.75 s (10k images) |
| **3** | **ANN** (Dense Fully-Connected) | Deep Learning | 97.38% | 97.36% | 34.6 s | 0.82 s (10k images) |
| **4** | **SVM** (RBF Kernel, C=10, $\gamma$=scale) | Traditional ML | 96.84% | 96.82% | 11.9 s (10k sub) | 37.66 s (10k images) |

---

## Running Locally

### 1. Install Dependencies
Ensure you are using Python 3.9 – 3.12:
```bash
pip install -r requirements.txt
```

### 2. Launch the Application
```bash
streamlit run app.py
```
Open **http://localhost:8501** in your browser.

---

## Deploying Online for Free (100% Free Cloud Hosting)

### Method 1: Streamlit Community Cloud (Recommended — 100% Free Forever)
1. Initialize a Git repository and push your project to GitHub:
   ```bash
   git init
   git add .
   git commit -m "Initial commit for digit recognition GUI"
   git branch -M main
   git remote add origin https://github.com/<your-username>/handwritten-digit-recognition.git
   git push -u origin main
   ```
2. Navigate to [share.streamlit.io](https://share.streamlit.io) and sign in using your GitHub account.
3. Click **"New app"**:
   - **Repository**: `<your-username>/handwritten-digit-recognition`
   - **Branch**: `main`
   - **Main file path**: `app.py`
4. Click **"Deploy!"**
5. Your app will be live with a permanent, shareable HTTPS URL (e.g., `https://handwritten-digit-recognition.streamlit.app`).

---

### Method 2: Hugging Face Spaces (100% Free Persistent Hosting)
1. Sign up or log in at [huggingface.co](https://huggingface.co).
2. Click **New Space** &rarr; Name your space &rarr; Select **Streamlit** SDK.
3. Choose the **Free CPU (2 vCPU, 16 GB RAM)** tier.
4. Clone the space repository locally or upload these files:
   - `app.py`
   - `best_model.keras`
   - `best_model_metadata.json`
   - `mnist_samples.npz`
   - `style.css`
   - `requirements.txt`
   - `.streamlit/config.toml`
5. Hugging Face will automatically install packages and deploy the app at `https://huggingface.co/spaces/<username>/<space-name>`.

---

## Preprocessing & Normalization Pipeline

To ensure real-world drawn digits match the distribution of the MNIST dataset, the application applies the exact NIST/MNIST standardization protocol:

```
User Input (Drawn Canvas / Uploaded Photo / MNIST Sample)
                           ↓
Convert to Grayscale (Luminance)
                           ↓
Polarity Detection (Ensure White Digit on Black Background)
                           ↓
Content Bounding Box Extraction
                           ↓
Aspect-Ratio Preserving Resize into 20×20 box (Lanczos Resampling)
                           ↓
Center on 28×28 Black Field
                           ↓
Center-of-Mass (Centroid) Alignment to (14, 14)
                           ↓
Pixel Intensity Normalization: [0, 255] → [0.0, 1.0]
                           ↓
Reshape to (1, 28, 28, 1) & Feed to Improved CNN
                           ↓
Softmax Probabilities & Predicted Class Output
```

---

## Repository Structure

```
├── app.py                         # Complete Streamlit GUI with Drawing & Upload
├── best_model.keras               # Winning trained model (Improved CNN, 99.06% Acc)
├── best_model_metadata.json       # Model metadata, parameters & benchmark metrics
├── mnist_samples.npz              # Bundled test samples for instant offline demonstration
├── style.css                      # Custom dark modern UI design
├── requirements.txt               # Production dependencies
├── .streamlit/
│   └── config.toml                # Streamlit server & theme configuration
└── DL_Handwritten_Digit_Recognition_Final.ipynb # Full academic notebook with comparative analysis
```

---

## Academic Viva Highlights

- **Convolution vs Dense (ANN)**: Dense layers fail to capture spatial invariance and destroy 2D topology; CNN filters slide locally to detect edges, curves, and digit loops.
- **Role of Batch Normalization**: Reduces internal covariate shift, allowing faster learning rates and stabilizing convergence.
- **Role of Dropout**: Randomly deactivates neurons during training to prevent co-dependency and overfitting.
- **Inference Latency Tradeoff**: SVM takes 37.66s to predict 10,000 samples due to kernel computations across thousands of support vectors ($O(N_{sv} \cdot D)$); CNN executes vectorized GPU/CPU matrix multiplications in 3.36s.
