# Amplifier_Health_Project
# COPD vs Healthy Classification

This project focuses on distinguishing between Chronic Obstructive Pulmonary Disease (COPD) patients and healthy individuals using audio recordings from the ICBHI 2017 dataset.

## Overview of Technique
The implemented approach involves manual feature extraction guided by insights from exploratory data analysis (EDA). Key features such as spectral properties, MFCCs, and statistical functionals were selected based on their relevance in representing respiratory sound characteristics. The extracted features were then processed using **SMOTE** (Synthetic Minority Oversampling Technique) to address class imbalance, ensuring better model generalization. An **XGBoost classifier** was trained on the balanced feature set, and hyperparameter optimization was performed to achieve the best classification performance.

## Running the Code
To execute the classification pipeline, use the following command:

```bash
python test.py
```

This command will:
1. Load preprocessed test data.
2. Extract features using the predefined pipeline.
3. Apply the trained XGBoost model to predict COPD or Healthy classes.
4. Output performance metrics, including classification reports and confusion matrices.

## Requirements
- Python 3.8 or higher
- Required libraries:
  - `numpy`
  - `pandas`
  - `opensmile`
  - `xgboost`
  - `imblearn`
  - `matplotlib`
  - `seaborn`

## Output
The results will include:
- A classification report with precision, recall, F1-score, and accuracy.
- A confusion matrix visualized as a heatmap.




