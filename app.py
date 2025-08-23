import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# Load model
model = joblib.load("salary_prediction_model.pkl")

# Page Config
st.set_page_config(page_title="💼 Salary Predictor", page_icon="💰", layout="wide")

# Custom CSS
st.markdown(
    """
    <style>
    body {
        background: linear-gradient(120deg, #fdfbfb 0%, #ebedee 100%);
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    .stApp {
        background: linear-gradient(120deg, #dfe9f3 0%, #ffffff 100%);
    }
    .title {
        text-align: center;
        font-size: 38px;
        font-weight: bold;
        color: #2c3e50;
        margin-bottom: 15px;
    }
    .subtitle {
        text-align: center;
        font-size: 18px;
        color: #7f8c8d;
        margin-bottom: 30px;
    }
    .input-card {
        background: white;
        padding: 20px;
        border-radius: 15px;
        box-shadow: 0px 4px 12px rgba(0,0,0,0.1);
        margin-bottom: 20px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Title
st.markdown("<div class='title'>💼 Salary Prediction App</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>Predict salaries and explore model insights</div>", unsafe_allow_html=True)

# Sidebar Inputs
st.sidebar.header("📊 Enter Features")
with st.sidebar:
    st.markdown("<div class='input-card'>", unsafe_allow_html=True)
    user_input = {
        "experience_level": st.selectbox("Experience Level", ["EN", "MI", "SE", "EX"]),
        "employment_type": st.selectbox("Employment Type", ["FT", "PT", "CT", "FL"]),
        "job_title": st.text_input("Job Title", "Data Scientist"),
        "company_size": st.selectbox("Company Size", ["S", "M", "L"]),
        "remote_ratio": st.slider("Remote Ratio (%)", 0, 100, 50),
        "company_location": st.text_input("Company Location", "US"),
        "employee_residence": st.text_input("Employee Residence", "US"),
    }
    st.markdown("</div>", unsafe_allow_html=True)

# Tabs
tab1, tab2, tab3 = st.tabs(["🔮 Prediction", "📈 Analytics", "ℹ️ About Model"])

with tab1:
    st.header("🔮 Salary Prediction")
    if st.button("🚀 Predict Salary"):
        input_df = pd.DataFrame([user_input])
        preds = model.predict(input_df)
        predicted_salary = f"${preds[0]:,.2f}"

        # Result Card
        st.markdown(
            f"""
            <div style="
                background: linear-gradient(90deg, #4facfe 0%, #00f2fe 100%);
                padding: 25px;
                border-radius: 15px;
                text-align: center;
                box-shadow: 0px 6px 15px rgba(0,0,0,0.25);
                margin-top: 20px;">
                <h2 style="color: white; font-size: 28px; margin: 0;">💰 Predicted Salary (USD)</h2>
                <h1 style="color: #fff; font-size: 46px; margin: 10px 0;">{predicted_salary}</h1>
            </div>
            """,
            unsafe_allow_html=True
        )

        st.balloons()
        st.snow()

with tab2:
    st.header("📊 Analytics & Insights")

    input_df = pd.DataFrame([user_input])
    preds = model.predict(input_df)

    # 📈 Scatter Plot
    st.subheader("Salary Distribution (Sample)")
    fig, ax = plt.subplots()
    sns.scatterplot(x=[i for i in range(len(input_df.columns))],
                    y=[preds[0] for _ in range(len(input_df.columns))],
                    s=120, color="blue", ax=ax)
    ax.set_title("Predicted Salary Scatter Plot")
    ax.set_xlabel("Features")
    ax.set_ylabel("Predicted Salary")
    st.pyplot(fig)

    # 🔑 Feature Importance
    if hasattr(model, "feature_importances_"):
        st.subheader("🔑 Feature Importance")
        importance = model.feature_importances_
        features = list(input_df.columns)
        fi_df = pd.DataFrame({"Feature": features, "Importance": importance})

        fig, ax = plt.subplots(figsize=(8, 5))
        sns.barplot(x="Importance", y="Feature", data=fi_df, palette="coolwarm", ax=ax)
        ax.set_title("Feature Importance")
        st.pyplot(fig)

with tab3:
    st.header("ℹ️ About the Model")
    st.markdown(
        """
        ### 🧠 Model Information
        - **Algorithm**: Gradient Boosting (LightGBM / Random Forest based)  
        - **Goal**: Predict salaries based on features such as experience level, employment type, job title, company size, location, and remote ratio.  
        - **Dataset**: Trained on a cleaned salary dataset.  
        - **Feature Engineering**: Categorical encoding + scaling applied.  

        ### 📌 Notes
        - Predictions are estimates based on historical patterns.  
        - Actual salaries may vary depending on negotiation, company budget, and market conditions.  
        """
    )
