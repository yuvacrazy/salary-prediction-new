import streamlit as st
import pandas as pd
import numpy as np
import lightgbm as lgb
import matplotlib.pyplot as plt
import seaborn as sns

# -----------------------------
# PAGE CONFIG & STYLING
# -----------------------------
st.set_page_config(page_title="Salary Prediction App", layout="wide")

st.markdown(
    """
    <style>
    body {
        background: linear-gradient(to right, #0f2027, #203a43, #2c5364);
        color: white;
        font-family: 'Segoe UI', sans-serif;
    }
    .big-font {
        font-size:32px !important;
        font-weight: bold;
        color: #FFD700;
    }
    .stButton>button {
        background: linear-gradient(90deg, #36D1DC, #5B86E5);
        color: white;
        border-radius: 12px;
        padding: 0.6em 1.2em;
        font-weight: bold;
        transition: 0.3s;
    }
    .stButton>button:hover {
        transform: scale(1.05);
        background: linear-gradient(90deg, #5B86E5, #36D1DC);
    }
    </style>
    """,
    unsafe_allow_html=True
)

# -----------------------------
# LOAD MODEL & DATA
# -----------------------------
model = lgb.Booster(model_file="salary_prediction_model.txt")

# Load dataset (optional for plots)
try:
    df = pd.read_csv("clean_salary_dataset.csv")
except:
    df = None

# -----------------------------
# MAIN APP
# -----------------------------
st.markdown("<p class='big-font'>💼 AI-Powered Salary Prediction</p>", unsafe_allow_html=True)

tab1, tab2, tab3 = st.tabs(["🔮 Prediction", "📊 Model Insights", "📈 Performance"])

# -----------------------------
# TAB 1: PREDICTION
# -----------------------------

with tab1:
    st.subheader("Enter Candidate Details")

    col1, col2, col3 = st.columns(3)
    with col1:
        age = st.number_input("Age", min_value=18, max_value=70, value=30)
    with col2:
        exp = st.number_input("Years of Experience", min_value=0, max_value=50, value=5)
    with col3:
        edu = st.selectbox("Education Level", ["Bachelor", "Master", "PhD"])

    education_map = {"Bachelor": 0, "Master": 1, "PhD": 2}

    if st.button("💰 Predict Salary"):
        X_new = np.array([[age, exp, education_map[edu]]])
        pred = model.predict(X_new)
        st.success(f"Estimated Salary: **${pred[0]:,.2f}**")

# -----------------------------
# TAB 2: FEATURE IMPORTANCE
# -----------------------------
with tab2:
    st.subheader("🔍 Feature Importance")

    if model:
        importance = model.feature_importance()
        feature_names = model.feature_name()   # <-- Dynamically fetch names

        fi_df = pd.DataFrame({
            "Feature": feature_names,
            "Importance": importance
        }).sort_values(by="Importance", ascending=True)

        fig, ax = plt.subplots(figsize=(6, 4))
        ax.barh(fi_df["Feature"], fi_df["Importance"], color="skyblue")
        ax.set_xlabel("Importance")
        ax.set_title("Feature Importance")
        st.pyplot(fig)

# -----------------------------
# TAB 3: PERFORMANCE
# -----------------------------
with tab3:
    st.subheader("📈 Model Performance")

    if df is not None:
        features = df.drop(columns=["Salary"])  # <-- dynamically use all features
        true_vals = df["Salary"]
        preds = model.predict(features)

        fig, ax = plt.subplots(figsize=(6,6))
        ax.scatter(true_vals, preds, alpha=0.6, color="orange", edgecolors="k")
        ax.set_xlabel("True Salary")
        ax.set_ylabel("Predicted Salary")
        ax.set_title("True vs Predicted Salaries")
        st.pyplot(fig)

        from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
        r2 = r2_score(true_vals, preds)
        mae = mean_absolute_error(true_vals, preds)
        rmse = mean_squared_error(true_vals, preds, squared=False)

        st.metric("R² Score", f"{r2:.3f}")
        st.metric("MAE", f"${mae:,.2f}")
        st.metric("RMSE", f"${rmse:,.2f}")

    else:
        st.warning("Dataset not found. Upload `clean_salary_dataset.csv` to see performance.")

