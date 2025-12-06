# Cybersecurity Intrusion Detection – Complete ML Pipeline

This repository contains a complete machine-learning workflow for analyzing, engineering features, and modeling a **cybersecurity intrusion detection dataset** using Python.  
The project includes:

- **EDA (Exploratory Data Analysis)**
- **Feature Importance & Statistical Insights**
- **Training Multiple ML Models** (RF, XGBoost, LightGBM, Logistic Regression)
- **Automated Hyperparameter Tuning (GridSearchCV)**
- **ROC Plotting & Metrics Summary**

---

##  Project Structure

```

dataAnalysis.py       # Full EDA: types, distributions, target balance, duplicates, encoding preview
numFeature.py         # Feature statistics: correlations, group-wise attack rates for features
model.py              # Complete ML pipeline: preprocessing, training, tuning, evaluation, ROC curves
cybersecurity_intrusion_data.csv   # Dataset 
README.md

````

---

##  Installation

Install required Python dependencies:

```bash
pip install pandas numpy scikit-learn xgboost lightgbm matplotlib
````

---

##  How to Run the Pipeline

### **Step 1 — Run EDA**

```bash
python dataAnalysis.py
```

### **Step 2 — Run Feature Analysis**

```bash
python numFeature.py
```

### **Step 3 — Train Machine Learning Models**

```bash
python model.py
```

This will:
 Do data analysis 
 train 4 ML models
 perform GridSearchCV hyperparameter tuning
 evaluate accuracy, F1, ROC-AUC
 report memory/time usage
 plot ROC curves for all models

---

#  Script Details

---

# 1️ `dataAnalysis.py` — Exploratory Data Analysis

This script performs the initial dataset investigation.

###  Operations Performed

* Prints **data types** for all features
* Shows **unique values** for categorical columns:

  * `session_id`
  * `protocol_type`
  * `encryption_used`
  * `browser_type`
* Displays **target balance** (`attack_detected`)
* Outputs **statistical summary** for all numeric features
* Shows frequency of:

  * failed login attempts
  * unusual time access
* Checks **duplicate session IDs**
* Demonstrates **example one-hot encoding** for `protocol_type`

###  Command

```bash
python dataAnalysis.py
```

---

# 2️ `numFeature.py` — Feature Importance & Statistical Insights

This script computes statistical relationships between features and the target.

###  Operations Performed

* Computes **correlation of every numeric feature** with the target (`attack_detected`)
* Computes **attack probability per protocol type**
* Computes attack probability for **all categorical columns** (except session_id)
* Useful for:

  * feature selection
  * understanding high-risk groups
  * building interpretable ML models

###  Command

```bash
python numFeature.py
```

---

# 3️ `model.py` — Complete Machine Learning Pipeline

This is the full modeling engine of the project.

###  Steps Performed

### **1. Load dataset**

Automatically loads `cybersecurity_intrusion_data.csv`.

### **2. Split Features**

* Numeric features → imputed & scaled
* Categorical features → imputed & one-hot encoded
* Removes `session_id` (non-predictive)

### **3. Train/Test Split**

```python
train_test_split(..., stratify=y)
```

### **4. Models Trained**

| Model                   | Description                |
| ----------------------- | -------------------------- |
| **Random Forest**       | Tree-based ensemble        |
| **XGBoost**             | Gradient boosted trees     |
| **LightGBM**            | High-performance boosting  |
| **Logistic Regression** | Linear baseline classifier |

Each model has its own **GridSearch hyperparameter tuning**.

### **5. Metrics Computed**

* Accuracy
* **F1 Score (primary metric)**
* ROC-AUC
* Training time
* Peak memory usage
* Estimated minimum required dataset size

### **6. Final Output**

* Best estimator per model
* Full metric summary printed
* **ROC curves plotted for all classifiers**

---

##  Example Output Metrics (printed automatically)

```
Random Forest Results:
Accuracy: 0.93
F1 Score: 0.91
ROC-AUC: 0.95
Training Time: 4.23s
Peak Memory: 220 MB
Min Data Points Needed: XXXX
```

---

##  Notes

* Ensure that your dataset file is named:
  **`cybersecurity_intrusion_data.csv`**
* XGBoost may require installing system libraries depending on OS.
* LightGBM may require:

  ```bash
  pip install lightgbm
  ```

---
