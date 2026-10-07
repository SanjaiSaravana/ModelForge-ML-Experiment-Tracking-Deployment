# 🔬 ModelForge — ML Experiment Tracking & Deployment

> **End-to-end ML experiment tracking with MLflow and one-click Gradio deployment**

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python)](https://python.org)
[![MLflow](https://img.shields.io/badge/MLflow-2.10%2B-orange?logo=mlflow)](https://mlflow.org)
[![XGBoost](https://img.shields.io/badge/XGBoost-2.0%2B-green)](https://xgboost.readthedocs.io)
[![Gradio](https://img.shields.io/badge/Gradio-5.0%2B-ff7c00?logo=gradio)](https://gradio.app)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## 📖 Overview

**ModelForge** is a complete, production-ready template for ML experiment tracking and model deployment. It demonstrates:

- 🧪 **Experiment Tracking** — Log parameters, metrics, and model artifacts with MLflow
- 📦 **Model Export** — Convert MLflow-tracked models to standalone `.pkl` files via joblib
- 🚀 **Instant Deployment** — Interactive Gradio web interface for real-time inference
- ☁️ **HuggingFace Ready** — Designed for seamless deployment on Hugging Face Spaces

---

## 🗂️ Project Structure

```
ModelForge/
├── app.py               # Gradio web application for model inference
├── mlflow_demo.py       # XGBoost training + MLflow experiment tracking
├── export_model.py      # Export trained model from MLflow → model.pkl
├── load_the_model.py    # Load model from MLflow and run predictions
├── requirements.txt     # Python dependencies
└── .gitignore           # Git ignore rules
```

---

## ⚙️ Setup & Installation

### 1. Clone the Repository

```bash
git clone https://github.com/SanjaiSaravana/ModelForge-ML-Experiment-Tracking-Deployment.git
cd ModelForge-ML-Experiment-Tracking-Deployment
```

### 2. Create a Virtual Environment

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 🚀 Usage

### Step 1 — Train the Model & Track with MLflow

```bash
python mlflow_demo.py
```

This will:
- Create a sample XGBoost classification dataset
- Train and evaluate the model
- Log parameters, metrics, and the model artifact to MLflow
- Export `model.pkl` for standalone use
- Save the run ID to `latest_run_id.txt`

**Sample Output:**
```
MLflow Experiment: Demo_Local_Run
Started MLflow Run ID: <run-id>
Model Training Complete. Test Accuracy: 1.0000
Model successfully logged to MLflow
```

### Step 2 — View Experiments in MLflow UI

```bash
mlflow ui
```

Open [http://localhost:5000](http://localhost:5000) in your browser to explore runs, compare metrics, and browse artifacts.

### Step 3 — Export Model (Optional)

If you need to re-export the model from MLflow to a standalone `.pkl`:

```bash
python export_model.py
# Or pass a specific run ID:
python export_model.py <run-id>
```

### Step 4 — Load & Test Predictions via MLflow

```bash
python load_the_model.py
# Or with a specific run ID:
python load_the_model.py <run-id>
```

### Step 5 — Launch the Gradio Web App

```bash
python app.py
```

Open [http://localhost:7860](http://localhost:7860) — enter **Feature 1** and **Feature 2** values and click **Predict**.

---

## 🌐 Gradio Interface

The web app provides:
- **Feature Inputs** — Numeric sliders for Feature 1 and Feature 2
- **Prediction Result** — Predicted class (0 or 1) with confidence probabilities
- **Sample Test Cases** — Pre-loaded examples for quick testing
- **Clear Button** — Reset inputs instantly

---

## 🧠 Model Details

| Property | Value |
|---|---|
| Algorithm | XGBoost Classifier |
| Features | `feature1` (1–10), `feature2` (11–20) |
| Target | Binary classification (Class 0 / Class 1) |
| Split | 80% train / 20% test |
| Metric | Accuracy |
| Tracking | MLflow (`Demo_Local_Run` experiment) |

---

## 📊 MLflow Tracked Parameters & Metrics

| Type | Name | Value |
|---|---|---|
| Parameter | `model_type` | XGBoost |
| Parameter | `test_size` | 0.2 |
| Parameter | `random_state` | 42 |
| Metric | `accuracy` | 1.0 |

---

## ☁️ Deployment on Hugging Face Spaces

1. Create a new Space at [huggingface.co/spaces](https://huggingface.co/spaces)
2. Set SDK to **Gradio**
3. Upload `app.py`, `model.pkl`, and `requirements.txt`
4. The space will automatically install dependencies and launch the app

> **Note:** `model.pkl` must be present before deploying. Run `mlflow_demo.py` or `export_model.py` locally first, then upload the generated file.

---

## 🛠️ Tech Stack

| Tool | Purpose |
|---|---|
| [MLflow](https://mlflow.org) | Experiment tracking, model registry, artifact storage |
| [XGBoost](https://xgboost.readthedocs.io) | Gradient boosted tree classifier |
| [scikit-learn](https://scikit-learn.org) | Train/test split, accuracy metrics |
| [Gradio](https://gradio.app) | Interactive web UI for model inference |
| [pandas](https://pandas.pydata.org) | Data manipulation and DataFrame creation |
| [joblib](https://joblib.readthedocs.io) | Model serialization to `.pkl` |

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).

---

## 🤝 Contributing

Contributions are welcome! Please open an issue or submit a pull request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

<p align="center">Made with ❤️ by <a href="https://github.com/SanjaiSaravana">SanjaiSaravana</a></p>
