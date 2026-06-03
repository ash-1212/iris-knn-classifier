# Project 2: Data Classification Using AI 🌸

## 👤 Intern Details
- **Name:** Ayesha Nazish
- **Organization:** DecodeLabs
- **Batch:** 2026
- **Project:** Project 2 — Data Classification Using AI

---

## 🎯 Project Objective
Build a complete supervised learning pipeline that trains a
K-Nearest Neighbors (KNN) classification model on the Iris
benchmark dataset — moving beyond rule-based logic into
real machine learning.

---

## 🔗 Connection to Project 1

| Project 1 (Rule-Based Chatbot) | Project 2 (ML Classifier) |
|---|---|
| Rules written manually | Rules learned from data |
| Dictionary lookup O(1) | KNN distance calculation |
| Deterministic output | Probabilistic prediction |
| No training needed | Requires training pipeline |
| if-else logic | Pattern recognition |

---

## 🧠 Architecture Used

This project follows the **IPO Model**:
INPUT                  PROCESS                OUTPUT
─────                  ───────                ──────
Iris Dataset     →     Train-Test Split   →   Confusion Matrix
Feature Scaling        KNN Algorithm          F1 Score
K Optimization         Predictions

---

## ✅ Features Implemented

| Feature | Status |
|---|---|
| Dataset loading and exploration | ✅ Done |
| Statistical summary of data | ✅ Done |
| Feature scaling (StandardScaler) | ✅ Done |
| Train-Test split 80/20 with stratify | ✅ Done |
| KNN model training (K=5) | ✅ Done |
| Predictions on unseen test data | ✅ Done |
| Confusion Matrix (text + heatmap) | ✅ Done |
| F1 Score evaluation | ✅ Done |
| Optimal K finder (elbow curve) | ✅ Done |
| Custom flower prediction function | ✅ Done |
| Feature analysis visualization | ✅ Done |

---

## 📁 Project Structure
Task2/
├── classifier.py        ← Main ML pipeline
├── confusion_matrix.png ← Model evaluation heatmap
├── k_optimization.png   ← Elbow curve chart
├── feature_analysis.png ← Feature analysis charts
└── README.md            ← This file

---

## 🚀 How to Run

### Requirements
- Python 3.7 or above
- Libraries: numpy, pandas, scikit-learn, matplotlib, seaborn

### Install Dependencies
```bash
pip install numpy pandas scikit-learn matplotlib seaborn
```

### Run the Classifier
```bash
cd e:\Decodelab_AI\Task2
python classifier.py
```

---

## 📊 Dataset Information

| Property | Value |
|---|---|
| Dataset | Iris Benchmark |
| Total Samples | 150 flowers |
| Features | 4 (Sepal L/W, Petal L/W) |
| Classes | 3 (Setosa, Versicolor, Virginica) |
| Balance | Perfectly balanced (50 per class) |
| Source | sklearn.datasets.load_iris() |

---

## 🤖 Model Information

| Property | Value |
|---|---|
| Algorithm | K-Nearest Neighbors (KNN) |
| K Value | 5 neighbors |
| Scaling | StandardScaler (mean=0, std=1) |
| Train Split | 120 samples (80%) |
| Test Split | 30 samples (20%) |
| Random State | 42 (reproducible) |
| Stratify | Yes (balanced classes) |

---

## 📈 Model Results

| Metric | Score |
|---|---|
| Training Accuracy | ~95-96% |
| Test Accuracy | ~96-97% |
| F1 Score (weighted) | ~0.9667 |
| Grade | EXCELLENT |

---

## 🔑 Key Concepts Demonstrated

- **Supervised Learning** — machine learns from labeled examples
- **Feature Scaling** — StandardScaler removes size bias from KNN
- **Train-Test Split** — honest evaluation on unseen data
- **KNN Algorithm** — proximity principle with majority voting
- **Confusion Matrix** — diagnostic tool showing WHERE errors occur
- **F1 Score** — honest metric that works beyond simple accuracy
- **K Optimization** — elbow curve to find best hyperparameter
- **Confidence Scores** — probability of each class prediction

---

## 💡 What I Learned

- How supervised learning differs from rule-based systems
- Why feature scaling is critical for distance-based algorithms
- How to prevent data leakage with proper train-test splits
- How to interpret confusion matrices and F1 scores
- Why accuracy alone can be misleading (accuracy mirage)
- How to tune hyperparameters using error rate curves
- How to deploy a model for real-world custom predictions

---

## 📊 Output Files Generated

| File | Description |
|---|---|
| `confusion_matrix.png` | Heatmap showing prediction accuracy per class |
| `k_optimization.png` | Elbow curve showing optimal K value |
| `feature_analysis.png` | Feature variance, species means, scatter plot |

---

*Built with ❤️ at DecodeLabs AI Industrial Training — Batch 2026*