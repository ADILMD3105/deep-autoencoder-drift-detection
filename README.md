# 🧠 Deep Autoencoder for Unsupervised Data Drift Detection

A deep learning-based system for detecting **data drift** in machine learning pipelines using a **Deep Autoencoder**. The model learns the normal behavior of baseline data and detects distribution shifts by analyzing reconstruction error without requiring labeled data.

---

## 📖 Overview

Machine learning models often assume that future data follows the same distribution as the data used during training. In real-world environments, however, data evolves over time due to changing user behavior, environmental conditions, or operational changes. This phenomenon, known as **data drift**, can significantly reduce model performance.

This project presents an **unsupervised drift detection system** built using PyTorch. A deep autoencoder is trained on baseline data to learn normal patterns. Incoming data is then reconstructed, and the reconstruction error is compared against a threshold derived from the baseline data. Samples exceeding the threshold are identified as potential drift.

---

## ✨ Features

- Deep Autoencoder implemented using PyTorch
- Fully unsupervised drift detection
- Automatic data preprocessing pipeline
- Reconstruction error-based anomaly detection
- 95th percentile threshold for drift detection
- Visualization of reconstruction error and drift
- Modular and easy-to-extend codebase

---

# 🔄 System Workflow

<p align="center">
<img src="images/workflow.png" width="850">
</p>

The dataset is first preprocessed and divided into baseline and stream data. The autoencoder is trained only on baseline data to learn normal patterns. Incoming stream data is reconstructed by the trained model, and reconstruction error is calculated. A statistical threshold is then used to determine whether the incoming data has drifted from the learned distribution.

---

# 🏗 Autoencoder Architecture

<p align="center">
<img src="images/architecture.png" width="650">
</p>

The encoder compresses the input into a lower-dimensional latent representation, while the decoder reconstructs the original input. Since the model has learned only normal data, unfamiliar patterns produce higher reconstruction errors, enabling drift detection.

---

# 📈 Training Performance

<p align="center">
<img src="images/training-loss.png" width="700">
</p>

The reconstruction loss decreases consistently during training, indicating that the model successfully learns the underlying characteristics of the baseline dataset.

---

# 🚨 Drift Detection

<p align="center">
<img src="images/drift-threshold.png" width="700">
</p>

A drift threshold is computed using the **95th percentile** of reconstruction errors from the baseline data. Incoming samples with reconstruction errors exceeding this threshold are classified as drifted data.

---

# 📊 Experimental Result

<p align="center">
<img src="images/creditcard-result.png" width="850">
</p>

The model successfully detects significant changes in incoming data by monitoring reconstruction error. Samples exceeding the drift threshold indicate distribution shifts that may negatively affect machine learning model performance.

---

# 🛠 Technologies Used

- Python
- PyTorch
- Pandas
- NumPy
- Scikit-learn
- Matplotlib

---

# 📂 Project Structure

```text
Deep-Autoencoder-Data-Drift-Detection
│
├── images/
├── src/
│   ├── autoencoder.py
│   ├── data_loader.py
│   ├── train.py
│   ├── detect_drift.py
│   └── drift_simulation.py
│
├── README.md
├── requirements.txt
├── LICENSE
└── .gitignore
```

---

# 🚀 Installation

Clone the repository

```bash
git clone https://github.com/yourusername/deep-autoencoder-data-drift-detection.git
```

Install dependencies

```bash
pip install -r requirements.txt
```

---

# ▶️ Usage

Train the model

```bash
python -m src.train path/to/dataset.csv
```

Run drift detection

```bash
python -m src.detect_drift path/to/dataset.csv
```

---

# 📁 Dataset

This project was evaluated using publicly available datasets including the **Credit Card Fraud Detection Dataset**, **Forest Cover Type Dataset (Covtype)**, and **Electricity Load Diagrams Dataset**.

Datasets are not included in this repository due to their size and licensing restrictions. Please download them from their official sources before running the project.

---

# 🔮 Future Improvements

- Online drift detection
- Adaptive threshold selection
- Feature-level drift localization
- Automatic model retraining
- Integration with MLOps pipelines

---
