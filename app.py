import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from catboost import CatBoostRegressor  # ✅ CatBoost instead of LightGBM

# -------------------------
# Load Model Safely
# -------------------------
try:
    model = joblib.load("salary_prediction_model.pkl")
except Exception as e:
    st.error(f"⚠️ Error loading model: {e}")
    st.stop()

# -------------------------
# Page Configuration
# -------------------------
st.set_page_config(page_title="Salary Prediction App", page_icon="💰", layout="wide")

st.markdown(
    """
    <style>
        body {
            background: linear-gradient(135deg, #2c3e50, #4ca1af);
            color: #fff;
        }
        .salary-card {
            background: linear-gradient(135deg, #00c6ff, #0072ff);
            padding: 25px;
            border-radius: 15px;
            text-align: center;
            font-size: 26px;
            font-weight: bold;
            color: white;
            box-shadow: 0px 4px 15px rgba(0,0,0,0.4);
        }
        h1, h2, h3 {
            color: #f8f9fa !important;
        }
    </style>
    """,
    unsafe_allow_html=True
)

st.title("💼 Salary Prediction App (CatBoost)")
st.markdown("Predict employee salaries based on job and experience features.")

# -------------------------
# Sidebar Inputs
# -------------------------
st.sidebar.header("User Input Features")

def user_input_features():
    work_year = st.sidebar.slider("Work Year", 2015, 2025, 2023)
    experience_level = st.sidebar.selectbox("Experience Level", ["EN", "MI", "SE", "EX"])
    employment_type = st.sidebar.selectbox("Employment Type", ["FT", "PT", "CT", "FL"])
    job_title = st.sidebar.text_input("Job Title", "Data Scientist")
    remote_ratio = st.sidebar.slider("Remote Ratio (%)", 0, 100, 50)
    company_size = st.sidebar.selectbox("Company Size", ["S", "M", "L"])
    company_location = st.sidebar.text_input("Company Location", "US")
    employee_residence = st.sidebar.text_input("Employee Residence", "US")

    data = {
        "work_year": work_year,
        "experience_level": experience_level,
        "employment_type": employment_type,
        "job_title": job_title,
        "remote_ratio": remote_ratio,
        "company_size": company_size,
        "company_location": company_location,
        "employee_residence": employee_residence,
    }
    return data

user_input = user_input_features()

# -------------------------
# Prediction
# -------------------------
if st.button("Predict Salary"):
    input_df = pd.DataFrame([user_input])
    try:
        preds = model.predict(input_df)
        st.markdown(
            f"""
            <div class="salary-card">
                💰 Predicted Salary (USD): ${preds[0]:,.2f}
            </div>
            """,
            unsafe_allow_html=True,
        )
    except Exception as e:
        st.error(f"⚠️ Prediction failed: {e}")

# -------------------------
# Feature Importance
# -------------------------
st.subheader("📊 Feature Importance")

try:
    df = pd.read_csv("clean_salary_dataset.csv")
    features = df.drop(columns=["salary_in_usd"])

    importance = model.get_feature_importance()
    feature_names = features.columns

    fi_df = pd.DataFrame({"Feature": feature_names, "Importance": importance})
    fi_df = fi_df.sort_values(by="Importance", ascending=False)

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.barplot(x="Importance", y="Feature", data=fi_df, ax=ax, palette="viridis")
    st.pyplot(fig)

except Exception as e:
    st.warning(f"Feature importance not available: {e}")
