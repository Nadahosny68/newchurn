### Churn Prediction App
🔗My Churn App Link: https://nadahosny69.streamlit.app/

## 🧠 App Purpose
The Churn Prediction App is an interactive web-based tool built using Streamlit. Based on simple demographic and usage data, it helps businesses predict whether a customer is likely to churn (leave) or stay.

## 🧩 Core Technologies Used
Tool / Library
Purpose
Streamlit
For creating the web UI
Pandas & NumPy
For data manipulation
Scikit-learn
For machine learning (Naive Bayes model)
Matplotlib & Seaborn
For data visualization
Excel/CSV Upload
To support both default and user-uploaded data


🚀 App Workflow
1. User Interface Setup
Page configured with a wide layout and title: "Customer Churn Prediction App"


Description markdown explains the app's goal.


2. Sidebar File Upload
User can upload either an Excel or CSV file.


If no file is uploaded, the app loads a default dataset using @st.cache_data.


3. Data Validation
The dataset must include: 'Age', 'Tenure', 'Sex', 'Churn'.


If columns are missing → app displays an error and stops.


Missing values are dropped.


'Sex' and 'Churn' values are mapped to numeric:


'Male' → 0, 'Female' → 1


'No' → 0, 'Yes' → 1


4. Model Training
Features used: Age, Tenure, Sex


Target: Churn


Split into training/testing sets (80%/20%)


Trained using Gaussian Naive Bayes (GaussianNB) from scikit-learn.


5. Main Page Visualizations
Data preview: shows the first few rows of the dataset.


Churn distribution chart: bar plot of how many customers churned vs. stayed.


6. Prediction Interface (Sidebar Input)
Sliders for:


Age (18–90)


Tenure (0–10 years)


Dropdown for:


Gender (Male or Female → mapped to 0 or 1)


Button: 🔮 Predict Churn


7. Prediction Output
Predicts Yes or No for churn.


Shows probability as:


Text output


Progress bar


Bar chart comparing churn vs. no churn probabilities


Shows model accuracy on test data.


8. Save Prediction
User can click “Save Result” to export the prediction result as a CSV file (prediction_result.csv), including:


Age


Tenure


Gender


Prediction (Yes/No)


Probability




📋 Required Columns in the Dataset
To function properly, the dataset must contain the following:
Age: Customer’s age (numeric)


Tenure: How long the customer has been with the company (numeric)


Sex: Gender – values must be Male or Female


Churn: Target value – must be Yes or No



💡 How to Use the App
Upload your customer dataset OR use the default one.


Review the dataset preview and churn distribution.


Input customer data using the sidebar sliders.


Click Predict Churn to view the result.


Optionally, click Save Result to export your prediction.



📈 Model Used
Algorithm: Gaussian Naive Bayes


Why this model:


Fast and simple


Works well with small and medium-sized datasets


Handles categorical → numeric conversions gracefully



🔚 Conclusion
This Churn Prediction App offers a clean, reliable, and easy-to-use interface for analyzing and predicting customer churn based on demographic data. It balances interactive input, real-time predictions, clean data handling, and error management—all within a single file.

