import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# App configuration
st.set_page_config(page_title="Churn Predictor", layout="wide")
st.title("📉 Customer Churn Prediction App")
st.markdown("Predict whether a customer is likely to churn based on basic demographics and tenure.")

# Load and preprocess data
@st.cache_data
def load_data():
    df = pd.read_excel("churn_dataset.xlsx")
    df['Sex'] = df['Sex'].map({'Male': 0, 'Female': 1})
    df['Churn'] = df['Churn'].map({'No': 0, 'Yes': 1})
    return df

df = load_data()

# Define features and labels
X = df[['Age', 'Tenure', 'Sex']]
y = df['Churn']

# Split and train model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = GaussianNB()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)

# === Layout === #
col1, col2 = st.columns([1, 1])

# Left Column: Dataset Preview
with col1:
    st.subheader("📄 Data Preview")
    st.dataframe(df.head(), use_container_width=True)

# Right Column: Churn Visualization
with col2:
    st.subheader("📊 Churn Distribution")
    fig, ax = plt.subplots()
    sns.countplot(data=df, x='Churn', palette='Set2', ax=ax)
    ax.set_xticklabels(['No Churn', 'Churn'])
    st.pyplot(fig)

# Sidebar for user input
st.sidebar.header("🧍 Enter Customer Data")
age = st.sidebar.slider("Age", 18, 90, 30)
tenure = st.sidebar.slider("Tenure (years)", 0, 10, 1)
gender = st.sidebar.selectbox("Gender", ['Male', 'Female'])
gender_value = 0 if gender == 'Male' else 1

if st.sidebar.button("🔮 Predict Churn"):
    input_data = np.array([[age, tenure, gender_value]])
    prediction = model.predict(input_data)[0]
    prob = model.predict_proba(input_data)[0]

    # Results section
    st.markdown("## 🔍 Prediction Result")
    st.success(f"**Churn Prediction:** {'Yes' if prediction == 1 else 'No'}")
    st.info(f"**Probability of Churn:** {prob[1]*100:.2f}%")
    st.write(f"📈 **Model Accuracy:** {acc*100:.2f}%")

    # Probability bar chart
    st.subheader("📉 Prediction Probability")
    fig2, ax2 = plt.subplots()
    ax2.bar(['No Churn', 'Churn'],
