# 🧠 Unsupervised Data Drift Detection using Deep Autoencoders

A deep learning-based system for detecting **data drift** in machine learning pipelines using a **Deep Autoencoder**. The model learns the normal characteristics of baseline data and identifies distribution shifts through reconstruction error analysis, enabling early detection of performance degradation without requiring labeled data.

---

## 📖 Overview

Machine learning models are generally trained on historical data with the assumption that future data will follow the same distribution. However, in real-world environments, data continuously evolves due to changes in user behavior, environmental conditions, system updates, or operational processes. This phenomenon is known as **data drift**, and if left undetected, it can significantly reduce the performance and reliability of deployed machine learning models.

This project presents an **unsupervised data drift detection system** built using **PyTorch**. A deep autoencoder is trained only on baseline (normal) data to learn its underlying patterns. During inference, incoming data is reconstructed by the trained model, and the reconstruction error is calculated for each sample. Samples producing unusually high reconstruction errors are identified as potential drift using a statistical threshold derived from baseline reconstruction errors.

The proposed system is designed to be dataset-independent and has been evaluated on multiple publicly available datasets to demonstrate its ability to detect both sudden and gradual data drift.

---

## ✨ Features

- Deep Autoencoder implemented using PyTorch
- Fully unsupervised data drift detection
- Automatic preprocessing pipeline for structured datasets
- Reconstruction error-based drift analysis
- Statistical threshold generation using the 95th percentile
- Visualization of model training and drift detection
- Modular and reusable project architecture
- Supports evaluation across multiple benchmark datasets

---

# 🔄 System Workflow

<p align="center">
  <img src="images/workflow.png" width="850">
</p>

### Workflow Summary

1. Load the input dataset.
2. Preprocess and normalize numerical features.
3. Split the dataset into **baseline** and **stream** data.
4. Train the Deep Autoencoder using only baseline data.
5. Reconstruct incoming stream samples.
6. Compute reconstruction error for each sample.
7. Calculate the drift threshold using the **95th percentile** of baseline reconstruction errors.
8. Flag samples exceeding the threshold as drifted.
9. Visualize reconstruction error and detected drift.

---

# 🏗 Deep Autoencoder Architecture

<p align="center">
  <img src="images/architecture.png" width="700">
</p>

The proposed model uses a fully connected deep autoencoder consisting of an encoder and decoder network.

- The **encoder** compresses the input into a lower-dimensional latent representation.
- The **decoder** reconstructs the original input from this compressed representation.
- Since the model is trained only on baseline data, it reconstructs familiar patterns accurately while producing larger reconstruction errors for unseen or drifted data.

This reconstruction error serves as the primary indicator for detecting distribution shifts.

---

# 📊 Experimental Results

The proposed approach was evaluated using publicly available benchmark datasets to verify its effectiveness in detecting data drift under different data distributions.

---

## 📉 Model Training

<p align="center">
  <img src="images/training-loss.png" width="700">
</p>

The reconstruction loss decreases steadily throughout training, demonstrating that the Deep Autoencoder successfully learns the normal characteristics of the baseline dataset.

---

## 🎯 Drift Threshold Generation

<p align="center">
  <img src="images/drift-threshold.png" width="700">
</p>

A drift threshold is automatically computed using the **95th percentile** of reconstruction errors obtained from the baseline data.

This adaptive threshold allows the model to distinguish normal reconstruction variations from significant distribution changes without requiring manually defined values.

---

## 📈 Drift Detection Result

<p align="center">
  <img src="images/creditcard-result.png" width="850">
</p>

The trained model successfully identifies samples whose reconstruction error exceeds the calculated threshold.

These samples are classified as **potential data drift**, indicating that the incoming data distribution differs significantly from the learned baseline distribution.

---

## ✅ Performance Summary

The proposed system successfully demonstrates:

- Learning normal data behavior without labeled samples.
- Detecting both sudden and gradual distribution shifts.
- Automatic threshold generation using statistical analysis.
- Generalization across multiple structured datasets.
- Simple and interpretable drift visualization.

---

# 🛠 Technology Stack

| Category | Technology |
|----------|------------|
| Programming Language | Python |
| Deep Learning | PyTorch |
| Data Processing | Pandas, NumPy |
| Machine Learning | Scikit-learn |
| Visualization | Matplotlib |

---

# 📂 Project Structure

```text
Unsupervised-Data-Drift-Detection
│
├── images/
│   ├── workflow.png
│   ├── architecture.png
│   ├── training-loss.png
│   ├── drift-threshold.png
│   └── creditcard-result.png
│
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

Clone the repository.

```bash
git clone https://github.com/yourusername/unsupervised-data-drift-detection.git
```

Install the required dependencies.

```bash
pip install -r requirements.txt
```

---

# ▶️ Usage

### Train the Autoencoder

```bash
python -m src.train path/to/dataset.csv
```

### Detect Data Drift

```bash
python -m src.detect_drift path/to/dataset.csv
```

---

# 📁 Datasets

The project was evaluated using publicly available datasets:

- Credit Card Fraud Detection Dataset
- Forest Cover Type (Covtype) Dataset
- Electricity Load Diagrams 2011–2014 Dataset

The datasets are **not included** in this repository due to their size and licensing considerations. Please download them from their official sources before running the project.

---

# 🔮 Future Improvements

- Feature-level drift localization
- Adaptive threshold optimization
- Online streaming data support
- Automatic model retraining
- Integration with MLOps monitoring pipelines
- Real-time dashboard for drift visualization

---

# 👨‍💻 Author

**Mohammed Adil**

Final Year B.Tech Computer Science Student

Interested in Machine Learning, Artificial Intelligence, and Backend Development.

---

## 📄 License

This project is licensed under the MIT License.
