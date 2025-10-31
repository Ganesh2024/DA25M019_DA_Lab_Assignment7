# DA5401 Assignment 7 — Multi-Class Model Selection using ROC and Precision-Recall Curves

## 📘 Overview

This repository contains my submission for **DA5401 (Foundations of Machine Learning) Assignment 7**, focusing on **multi-class model selection using ROC and Precision-Recall curves**.

The goal is to evaluate and compare multiple classification models on the **UCI Landsat Satellite dataset**, analyzing their performance not just through accuracy, but by interpreting **ROC (Receiver Operating Characteristic)** and **Precision-Recall Curves (PRC)** across multiple classes.

---

## 🎯 Objective

To identify the best-performing and worst-performing classifiers for a **6-class satellite land cover classification** problem using **One-vs-Rest (OvR)** multi-class ROC and PRC analysis.

Key evaluation aspects:
- Understanding **AUC (Area Under Curve)** and **Average Precision (AP)** in a multi-class context.  
- Visualizing **macro-averaged ROC** and **PRC curves** for each model.
- Comparing performance across models to recommend the best classifier.

---

## 🧩 Dataset

**Dataset:** [UCI Landsat Satellite Dataset](https://archive.ics.uci.edu/ml/datasets/Statlog+(Landsat+Satellite))  
**Classes:** 6 land cover types  
**Features:** Multispectral values from satellite images  
**Preprocessing:**
- Loaded and cleaned data
- Standardized features using `StandardScaler`
- Split into **train** and **test** sets

---

## ⚙️ Models Implemented

| Model | Library | Expected Performance |
|--------|----------|----------------------|
| K-Nearest Neighbors (KNN) | `sklearn.neighbors.KNeighborsClassifier` | Moderate / Good |
| Decision Tree Classifier | `sklearn.tree.DecisionTreeClassifier` | Moderate |
| Dummy Classifier (Prior) | `sklearn.dummy.DummyClassifier` | Baseline / Poor (AUC < 0.5) |
| Logistic Regression | `sklearn.linear_model.LogisticRegression` | Good (Linear baseline) |
| Gaussian Naive Bayes | `sklearn.naive_bayes.GaussianNB` | Variable / Often poor |
| Support Vector Machine (SVC) | `sklearn.svm.SVC` | Good (with `probability=True`) |

**Bonus Models (Brownie Points):**
- Random Forest Classifier
- XGBoost Classifier
- One additional poor model (AUC < 0.5) for analysis

---

## 🧪 Experiments and Analysis

### **Part A: Data Preparation and Baseline**
- Preprocessed and standardized the dataset.
- Trained all six models using an 80–20 train-test split.
- Computed **Overall Accuracy** and **Weighted F1-Score** as initial baselines.

### **Part B: Multi-Class ROC Analysis**
- Used **One-vs-Rest** strategy to compute ROC curves for all classes.
- Generated a **combined ROC plot** with macro-averaged AUC for all models.
- Identified:
  - Model with **highest AUC**
  - Model with **AUC < 0.5**, interpreting what this means conceptually

### **Part C: Precision-Recall Curve (PRC) Analysis**
- Computed **macro-averaged Precision-Recall Curves** for each model.
- Discussed why PRC is more suitable than ROC when class imbalance exists.
- Identified:
  - Model with **highest Average Precision (AP)**
  - Behavior of **poor models** where PR drops sharply as recall increases

### **Part D: Final Recommendation**
- Compared rankings from F1-score, ROC-AUC, and PRC-AP.
- Discussed trade-offs between models (e.g., high ROC-AUC but lower PRC-AP).
- Recommended the most balanced model for multi-class performance.

---

## 📊 Visualization Highlights

- **Combined ROC Plot:** All models compared on one figure for macro-averaged ROC.
- **Combined PRC Plot:** Macro-averaged PRC for the same models.
- **Interpretation Sections:** Explained curve behaviors and threshold-based trade-offs.

---

## 🧠 Key Learnings

- **ROC curves** can be misleading in multi-class or imbalanced problems — PRC provides deeper insight.
- **Dummy classifiers** can yield AUC < 0.5 for minority classes, representing models worse than random guessing.
- Ensemble models like **Random Forest** and **XGBoost** often outperform simple baselines, but interpretation matters more than raw accuracy.
- Model selection should consider **precision-recall balance**, not just AUC.

---

## 🧾 File Structure

```
.
├── Assignment7.ipynb        # Jupyter Notebook with complete code and analysis
├── DA5401 A7 Model Selection.pdf  # Original assignment description
└── README.md                # This file (project overview)
```

---

## 🚀 How to Run

1. Clone this repository:
   ```bash
   git clone https://github.com/<your-username>/DA5401-A7-Model-Selection.git
   cd DA5401-A7-Model-Selection
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Open and run the notebook:
   ```bash
   jupyter notebook Assignment7.ipynb
   ```

---

## 🧰 Dependencies

```txt
numpy
pandas
matplotlib
seaborn
scikit-learn
xgboost
```

---

## 🧩 Author

**Ganesh**  
M.Tech Student — Data Science & Artificial Intelligence  
Indian Institute of Technology Madras  

---

## 🏁 Acknowledgment

This project was completed as part of **DA5401: Foundations of Machine Learning**, taught by **Dr. Arun**.  
Special thanks to the course team for providing clear instructions and the dataset reference.

---
