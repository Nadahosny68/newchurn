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
st.title("Customer Churn Prediction App")
st.markdown("Predict whether a customer is likely to churn based on basic demographics and tenure.")




# Load default or user-uploaded dataset
@st.cache_data
def load_default_data():
    return pd.read_excel("churn_dataset.xlsx")

st.sidebar.header("📁 Upload Dataset")
uploaded_file = st.sidebar.file_uploader("Upload an Excel or CSV file", type=["xlsx", "csv"])

if uploaded_file is not None:
    try:
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)
        st.success("✅ File uploaded and loaded successfully!")
    except Exception as e:
        st.error(f"❌ Error loading file: {e}")
        df = load_default_data()
else:
    df = load_default_data()
    st.info("ℹ️ Using default dataset.")




# Data preprocessing
required_columns = ['Age', 'Tenure', 'Sex', 'Churn']

if not all(col in df.columns for col in required_columns):
    st.error(f"❌ The uploaded file must contain the following columns: {required_columns}")
    st.stop()

# Drop rows with missing values in required columns
df = df[required_columns].dropna()

# Map categorical values
df['Sex'] = df['Sex'].map({'Male': 0, 'Female': 1})
df['Churn'] = df['Churn'].map({'No': 0, 'Yes': 1})

# Ensure numeric types
df[['Age', 'Tenure', 'Sex', 'Churn']] = df[['Age', 'Tenure', 'Sex', 'Churn']].apply(pd.to_numeric, errors='coerce')
df.dropna(inplace=True)

# Features and labels
X = df[['Age', 'Tenure', 'Sex']]
y = df['Churn']

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = GaussianNB()
model.fit(X_train, y_train)









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

    # Display Prediction as a Progress Bar
    st.subheader("📈 Churn Probability")
    st.progress(int(prob[1] * 100))


    # Probability bar chart
    st.subheader("📉 Prediction Probability")
    fig2, ax2 = plt.subplots()
    ax2.bar(['No Churn', 'Churn'], prob, color=['green', 'red'])  # Add 'prob' as height
    ax2.set_ylabel("Probability")
    ax2.set_ylim(0, 1)  # Optional: sets y-axis from 0 to 1
    st.pyplot(fig2)

if st.sidebar.button("Save Result"):
    result = {
        'Age': age,
        'Tenure': tenure,
        'Gender': gender,
        'Prediction': 'Yes' if prediction == 1 else 'No',
        'Probability': prob[1]
    }
    results_df = pd.DataFrame([result])
    results_df.to_csv("prediction_result.csv", index=False)
    st.success("Result saved to prediction_result.csv")

#Add File Upload for Dataset (Optional override
uploaded_file = st.sidebar.file_uploader("Upload your own Excel file", type=["xlsx"])
if uploaded_file:
    df = pd.read_excel(uploaded_file)
else:
    df = load_data()
