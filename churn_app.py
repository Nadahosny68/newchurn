import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, confusion_matrix

# Page title
st.title("📊 Customer Churn Prediction App")

# Load the dataset
@st.cache_data
def load_data():
    url = "https://raw.githubusercontent.com/hosnynada69/churn-data/main/churn_dataset_with_tenure.csv"
    df = pd.read_csv(url)
    return df

df = load_data()

# Data preprocessing
df['Sex'] = df['Sex'].map({'Male': 0, 'Female': 1})
df['Churn'] = df['Churn'].map({'No': 0, 'Yes': 1})

# Features and labels
X = df[['Age', 'Tenure', 'Sex']]
y = df['Churn']

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = GaussianNB()
model.fit(X_train, y_train)

# Accuracy
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)

# Sidebar user input
st.sidebar.header("📥 Enter Customer Data")
age = st.sidebar.slider("Age", 18, 90, 30)
tenure = st.sidebar.slider("Tenure (years)", 0, 10, 1)
gender = st.sidebar.selectbox("Gender", ['Male', 'Female'])
gender_value = 0 if gender == 'Male' else 1

# Prediction
if st.sidebar.button("Predict"):
    input_data = np.array([[age, tenure, gender_value]])
    prediction = model.predict(input_data)[0]
    prob = model.predict_proba(input_data)[0]

    st.subheader("🔍 Prediction Result:")
    st.write(f"**Churn Prediction:** {'Yes' if prediction == 1 else 'No'}")
    st.write(f"**Probability of Churn:** {prob[1]*100:.2f}%")
    st.write(f"**Model Accuracy:** {acc*100:.2f}%")

    # Confusion matrix (Bonus)
    st.subheader("🧮 Confusion Matrix")
    cm = confusion_matrix(y_test, y_pred)
    st.write(pd.DataFrame(cm, columns=["Predicted No", "Predicted Yes"], index=["Actual No", "Actual Yes"]))
