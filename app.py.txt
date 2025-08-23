import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

# ---------------- Load Model ---------------- #
model = joblib.load("salary_prediction_model.pkl")

# ---------------- Page Config ---------------- #
st.set_page_config(page_title="💼 Salary Prediction Dashboard", page_icon="💰", layout="wide")

# ---------------- Custom CSS ---------------- #
st.markdown("""
    <style>
    /* Background gradient */
    .stApp {
        background: linear-gradient(135deg, #1e3c72, #2a5298);
        color: #ffffff;
    }

    /* Card Style */
    .prediction-card, .metric-card {
        background: rgba(255, 255, 255, 0.1);
        padding: 20px;
        border-radius: 16px;
        box-shadow: 0 4px 30px rgba(0,0,0,0.2);
        backdrop-filter: blur(10px);
        color: #fff;
    }

    h1, h2, h3, h4 {
        color: #f1f1f1 !important;
    }

    .stTabs [role="tablist"] {
        justify-content: center;
    }

    .stTabs [role="tab"] {
        background: rgba(255,255,255,0.1);
        border-radius: 12px;
        margin: 0 5px;
        color: #fff !important;
    }
    </style>
""", unsafe_allow_html=True)

# ---------------- Title ---------------- #
st.markdown("<h1 style='text-align: center;'>💼 AI-Powered Salary Prediction</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>An interactive dashboard to estimate salaries, analyze features, and evaluate model performance.</p>", unsafe_allow_html=True)

# ---------------- Tabs ---------------- #
tab1, tab2, tab3 = st.tabs(["🔮 Prediction", "📊 Insights", "📈 Performance"])

# ---------------- Tab 1: Prediction ---------------- #
with tab1:
    st.markdown("### 📋 Enter Job Information")

    with st.form("salary_form"):
        col1, col2 = st.columns(2)

        with col1:
            experience_level = st.selectbox("Experience Level", ["Entry-level", "Mid-level", "Senior", "Executive"])
            employment_type = st.selectbox("Employment Type", ["FT", "PT", "Contract", "Freelance"])
            job_title = st.text_input("Job Title", "Data Scientist")

        with col2:
            employee_residence = st.text_input("Employee Residence (e.g., US, IN, UK)", "US")
            company_location = st.text_input("Company Location (e.g., US, IN, UK)", "US")
            company_size = st.selectbox("Company Size", ["S", "M", "L"])

        remote_ratio = st.slider("Remote Work Ratio (%)", 0, 100, 50)

        submitted = st.form_submit_button("🚀 Predict Salary")

    if submitted:
        input_data = pd.DataFrame([{
            "experience_level": experience_level,
            "employment_type": employment_type,
            "job_title": job_title,
            "employee_residence": employee_residence,
            "company_location": company_location,
            "remote_ratio": remote_ratio,
            "company_size": company_size
        }])

        prediction = model.predict(input_data)[0]

        st.markdown("<div class='prediction-card'>", unsafe_allow_html=True)
        st.markdown(f"<h2 style='text-align:center;'>💰 Estimated Salary</h2>", unsafe_allow_html=True)
        st.markdown(f"<h1 style='text-align:center; color:#00FFB2;'>${prediction:,.2f}</h1>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

# ---------------- Tab 2: Insights (Feature Importance) ---------------- #
with tab2:
    st.markdown("### 📊 Feature Importance")

    lgbm_model = model.named_steps["regressor"]
    encoded_features = model.named_steps["preprocessor"].get_feature_names_out()
    importances = lgbm_model.feature_importances_

    indices = np.argsort(importances)[::-1][:15]
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(range(len(indices)), importances[indices], align="center", color="#00FFB2")
    ax.set_yticks(range(len(indices)))
    ax.set_yticklabels([encoded_features[i] for i in indices], color="white")
    ax.invert_yaxis()
    ax.set_xlabel("Feature Importance", color="white")
    ax.set_title("Top 15 Features Driving Salary Prediction", color="white")
    fig.patch.set_alpha(0)
    ax.patch.set_alpha(0)
    st.pyplot(fig)

    st.markdown("<p style='text-align:center; color:#aaa;'>💡 Higher importance = stronger influence on salary prediction</p>", unsafe_allow_html=True)

# ---------------- Tab 3: Performance ---------------- #
with tab3:
    st.markdown("### 📈 Model Performance")

    # Reload dataset for evaluation
    df = pd.read_csv("clean_salary_dataset.csv").dropna(subset=["salary_in_usd"])
    df = df.rename(columns={"salary_in_usd": "salary"})
    relevant_features = [
        "experience_level", "employment_type", "job_title",
        "employee_residence", "company_location", "remote_ratio",
        "company_size", "salary"
    ]
    df = df[relevant_features].dropna().reset_index(drop=True)

    X = df.drop("salary", axis=1)
    y = df["salary"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    y_pred = model.predict(X_test)

    # Metrics
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)
    col1.metric("📏 R² Score", f"{r2:.3f}")
    col2.metric("📉 MAE", f"{mae:,.2f}")
    col3.metric("📊 RMSE", f"{rmse:,.2f}")
    st.markdown("</div>", unsafe_allow_html=True)

    # Scatter Plot
    st.markdown("### 🔍 Actual vs Predicted Salaries")
    fig2, ax2 = plt.subplots(figsize=(7, 6))
    ax2.scatter(y_test, y_pred, alpha=0.4, color="#00FFB2", edgecolors="white")
    ax2.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "r--", lw=2)
    ax2.set_xlabel("Actual Salary", color="white")
    ax2.set_ylabel("Predicted Salary", color="white")
    ax2.set_title("Actual vs Predicted Salary", color="white")
    fig2.patch.set_alpha(0)
    ax2.patch.set_alpha(0)
    st.pyplot(fig2)
