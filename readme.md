# Bayesian Explainable AI for Alzheimer Risk Assessment
### Research & Development
**Naina**

---
## Overview

This project presents a trustworthy and explainable healthcare AI framework for early Alzheimer disease risk assessment using machine learning, explainable AI, concept-aware reasoning, and Bayesian-inspired uncertainty estimation.

The system is designed as a clinical decision-support prototype capable of:
- predicting Alzheimer risk
- estimating prediction confidence
- explaining AI decisions
- generating counterfactual recommendations
- identifying intermediate clinical concepts
- supporting interpretable healthcare AI research

---

# Key Features

## Explainable Machine Learning
- XGBoost-based Alzheimer prediction
- Hyperparameter optimization
- Leakage-aware healthcare modeling
- Class imbalance handling

## Bayesian-Inspired Uncertainty Estimation
- Ensemble-based uncertainty analysis
- Confidence-aware prediction
- Ambiguous patient identification

## SHAP Explainability
- Global feature importance
- Local patient-level explanations
- Transparent clinical interpretation

## Counterfactual AI
- DiCE-based recommendation generation
- Suggests modifiable risk-factor improvements

## Concept-Aware Clinical Reasoning
Intermediate clinical concepts:
- Cognitive Decline
- Lifestyle Risk
- Cardiovascular Risk
- Functional Decline

## Clinical Decision Support System
Integrates:
- prediction
- uncertainty
- explainability
- recommendations
- concept reasoning

into a unified healthcare AI framework.

## Streamlit Deployment
Interactive clinical dashboard for:
- patient data input
- Alzheimer risk prediction
- explainability visualization
- uncertainty reporting
- recommendation generation

---

# Dataset Features

## Personal Information
- Age
- Gender
- Ethnicity
- Education Level

## Lifestyle Factors
- Smoking
- Alcohol Consumption
- Physical Activity
- Diet Quality
- Sleep Quality

## Medical Conditions
- Diabetes
- Hypertension
- Cardiovascular Disease
- Depression
- Head Injury

## Clinical Assessments
- MMSE
- Functional Assessment
- ADL

## Laboratory Measurements
- Cholesterol
- LDL
- HDL
- Triglycerides
- Blood Pressure

---

# Machine Learning Pipeline

```text
Patient Data
↓
Data Cleaning
↓
Leakage Removal
↓
XGBoost Prediction
↓
Hyperparameter Optimization
↓
Confidence Calibration
↓
SHAP Explainability
↓
Counterfactual AI
↓
Concept Bottleneck Layer
↓
Bayesian Uncertainty Estimation
↓
Clinical Decision Support AI
↓
Streamlit Deployment
```

---

# Technologies Used

- Python
- XGBoost
- Scikit-learn
- SHAP
- DiCE-ML
- Streamlit
- Pandas
- NumPy
- Matplotlib
- Plotly

---

# Installation

## Clone Repository

```bash
git clone https://github.com/your-username/repository-name.git
```

## Install Dependencies

```bash
pip install -r requirements.txt
```

## Run Streamlit Application

```bash
streamlit run app.py
```

---

# Project Structure

```text
project/
│
├── app.py
├── requirements.txt
├── README.md
├── alzheimers_xgboost_model.pkl
│
├── dataset/
│   └── alzheimers_disease_data.csv
│
├── notebooks/
│   └── model_training.ipynb
│
└── outputs/
    ├── shap_plots/
    ├── confusion_matrix/
    └── reports/
```

---

# Research Contributions

This project focuses on developing an interpretable and trustworthy healthcare AI framework rather than only maximizing prediction accuracy.

Key contributions include:
- Leakage-aware clinical modeling
- Bayesian-inspired uncertainty estimation
- Explainable healthcare AI
- Counterfactual recommendation generation
- Concept-aware clinical reasoning
- Clinical decision-support integration

---

# Future Improvements

- MRI-based multimodal learning
- Deep Concept Bottleneck Networks
- External clinical validation
- Federated healthcare learning
- Real-time hospital deployment
- Longitudinal disease progression analysis

---

# Disclaimer

This project is intended for educational and research purposes only and should not be used as a replacement for professional medical diagnosis.
---

# License

This project is licensed under the  
Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International License (CC BY-NC-ND 4.0).

© 2026 Naina

Users may view and share this work with proper attribution, but may not:
- use it commercially
- modify and redistribute it
- claim authorship of the original work