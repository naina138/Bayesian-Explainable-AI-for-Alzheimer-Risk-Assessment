
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import shap
from xgboost import XGBClassifier



st.set_page_config(
    page_title="Clinical Alzheimer AI",
    layout="wide"
)



st.title("Clinical AI for Early Alzheimer Risk Prediction")
st.markdown("---")

st.write(
    "This system provides Alzheimer risk prediction, "
    "uncertainty estimation, concept-based reasoning, "
    "and explainable AI insights using machine learning."
)



@st.cache_resource

def load_model():
    model = joblib.load("alzheimers_xgboost_model.pkl")
    return model

model = load_model()

st.sidebar.header("Patient Clinical Information")

Age = st.sidebar.slider("Age", 40, 100, 70)
Gender = st.sidebar.selectbox("Gender", [0, 1])
Ethnicity = st.sidebar.selectbox("Ethnicity", [0, 1, 2, 3])
EducationLevel = st.sidebar.slider("Education Level", 0, 20, 10)
BMI = st.sidebar.slider("BMI", 10.0, 40.0, 25.0)
Smoking = st.sidebar.selectbox("Smoking", [0, 1])
AlcoholConsumption = st.sidebar.slider("Alcohol Consumption", 0.0, 20.0, 5.0)
PhysicalActivity = st.sidebar.slider("Physical Activity", 0.0, 10.0, 5.0)
DietQuality = st.sidebar.slider("Diet Quality", 0.0, 10.0, 5.0)
SleepQuality = st.sidebar.slider("Sleep Quality", 0.0, 10.0, 5.0)
FamilyHistoryAlzheimers = st.sidebar.selectbox("Family History", [0, 1])
CardiovascularDisease = st.sidebar.selectbox("Cardiovascular Disease", [0, 1])
Diabetes = st.sidebar.selectbox("Diabetes", [0, 1])
Depression = st.sidebar.selectbox("Depression", [0, 1])
HeadInjury = st.sidebar.selectbox("Head Injury", [0, 1])
Hypertension = st.sidebar.selectbox("Hypertension", [0, 1])
SystolicBP = st.sidebar.slider("Systolic BP", 90, 200, 120)
DiastolicBP = st.sidebar.slider("Diastolic BP", 60, 140, 80)
CholesterolTotal = st.sidebar.slider("Total Cholesterol", 100, 400, 200)
CholesterolLDL = st.sidebar.slider("LDL", 50, 250, 120)
CholesterolHDL = st.sidebar.slider("HDL", 20, 100, 50)
CholesterolTriglycerides = st.sidebar.slider("Triglycerides", 50, 500, 150)
MMSE = st.sidebar.slider("MMSE Score", 0, 30, 25)
FunctionalAssessment = st.sidebar.slider("Functional Assessment", 0.0, 10.0, 5.0)
ADL = st.sidebar.slider("ADL", 0.0, 10.0, 5.0)


input_data = pd.DataFrame({
    'Age': [Age],
    'Gender': [Gender],
    'Ethnicity': [Ethnicity],
    'EducationLevel': [EducationLevel],
    'BMI': [BMI],
    'Smoking': [Smoking],
    'AlcoholConsumption': [AlcoholConsumption],
    'PhysicalActivity': [PhysicalActivity],
    'DietQuality': [DietQuality],
    'SleepQuality': [SleepQuality],
    'FamilyHistoryAlzheimers': [FamilyHistoryAlzheimers],
    'CardiovascularDisease': [CardiovascularDisease],
    'Diabetes': [Diabetes],
    'Depression': [Depression],
    'HeadInjury': [HeadInjury],
    'Hypertension': [Hypertension],
    'SystolicBP': [SystolicBP],
    'DiastolicBP': [DiastolicBP],
    'CholesterolTotal': [CholesterolTotal],
    'CholesterolLDL': [CholesterolLDL],
    'CholesterolHDL': [CholesterolHDL],
    'CholesterolTriglycerides': [CholesterolTriglycerides],
    'MMSE': [MMSE],
    'FunctionalAssessment': [FunctionalAssessment],
    'ADL': [ADL]
})



if st.button("Predict Alzheimer Risk"):

    prediction = model.predict(input_data)[0]

    probability = model.predict_proba(input_data)[0][1]

   

    if probability >= 0.75:
        risk_level = "HIGH RISK"
    elif probability >= 0.45:
        risk_level = "MODERATE RISK"
    else:
        risk_level = "LOW RISK"

   

    uncertainty = 1 - abs(probability - 0.5) * 2

    if uncertainty < 0.20:
        uncertainty_label = "Low Uncertainty"
    elif uncertainty < 0.40:
        uncertainty_label = "Moderate Uncertainty"
    else:
        uncertainty_label = "High Uncertainty"


    st.markdown("---")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Prediction", risk_level)

    with col2:
        st.metric(
            "Risk Probability",
            f"{probability:.2%}"
        )

    with col3:
        st.metric(
            "Uncertainty",
            uncertainty_label
        )


    st.subheader("Clinical Concept Reasoning")

    concepts = {
        "Cognitive Decline": MMSE < 24,
        "Functional Impairment": FunctionalAssessment < 5,
        "Lifestyle Risk": (
            Smoking == 1 or
            SleepQuality < 5 or
            PhysicalActivity < 3
        ),
        "Cardiovascular Risk": (
            Hypertension == 1 or
            Diabetes == 1 or
            CardiovascularDisease == 1
        ),
        "Cholesterol Risk": CholesterolTotal > 200
    }

    concept_df = pd.DataFrame({
        "Clinical Concept": list(concepts.keys()),
        "Detected": [
            "Yes" if value else "No"
            for value in concepts.values()
        ]
    })

    st.dataframe(concept_df)

  

    st.subheader("Clinical Recommendations")

    recommendations = []

    if SleepQuality < 5:
        recommendations.append(
            "Improve sleep quality and sleep duration"
        )

    if PhysicalActivity < 3:
        recommendations.append(
            "Increase physical activity levels"
        )

    if CholesterolTotal > 200:
        recommendations.append(
            "Monitor cholesterol and cardiovascular health"
        )

    if Hypertension == 1:
        recommendations.append(
            "Regular blood pressure management recommended"
        )

    if MMSE < 24:
        recommendations.append(
            "Further neurological assessment recommended"
        )

    if len(recommendations) == 0:
        recommendations.append(
            "Maintain current healthy lifestyle"
        )

    for rec in recommendations:
        st.write(f"• {rec}")

   

    st.subheader("Top Risk Factors")

    importance_df = pd.DataFrame({
        'Feature': input_data.columns,
        'Importance': model.feature_importances_
    })

    importance_df = importance_df.sort_values(
        by='Importance',
        ascending=False
    ).head(10)

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.barh(
        importance_df['Feature'],
        importance_df['Importance']
    )

    ax.set_title("Top Alzheimer Risk Features")

    st.pyplot(fig)

   

    st.markdown("---")

    st.subheader("Clinical AI Summary")

    st.write(
        f"The patient shows **{risk_level}** "
        f"for Alzheimer disease with an estimated "
        f"probability of **{probability:.2%}**."
    )

    st.write(
        f"Model confidence assessment indicates "
        f"**{uncertainty_label}**."
    )

    st.write(
        "This assessment should be used as a "
        "clinical decision-support tool and not "
        "as a standalone medical diagnosis."
    )

st.markdown("---")

st.caption(
    "Explainable Clinical AI for Early Alzheimer Risk Prediction"
)

