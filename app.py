from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)


folder = Path(__file__).resolve().parent
model_files = {
    "Logistic Regression": "logistic_regression.joblib",
    "Decision Tree": "decision_tree.joblib",
    "KNN": "knn.joblib",
    "Naive Bayes": "naive_bayes.joblib",
    "Support Vector Machine": "svm.joblib",
}


@st.cache_resource
def load_model(filename):
    return joblib.load(folder / "model" / filename)


st.title("Email Spam Classification")
st.write("Upload test data with all input columns and the spam column: 0 = not spam, 1 = spam.")
model_name = st.selectbox("Choose a model", list(model_files))
uploaded_file = st.file_uploader("Upload test CSV", type="csv")

if uploaded_file is None:
    st.info("Upload test_data.csv to see the results.")
    st.stop()

try:
    data = pd.read_csv(uploaded_file)
except (pd.errors.EmptyDataError, pd.errors.ParserError, UnicodeDecodeError):
    st.error("The file could not be read. Upload a valid CSV file.")
    st.stop()

if data.empty:
    st.error("The CSV has no data rows.")
    st.stop()

try:
    model = load_model(model_files[model_name])
except FileNotFoundError:
    st.error("The saved model is missing. Run the training notebook first.")
    st.stop()

features = list(model.feature_names_in_)
required_columns = features + ["spam"]
missing = [column for column in required_columns if column not in data.columns]
extra = [column for column in data.columns if column not in required_columns]
if missing or extra:
    if missing:
        st.error("Missing columns: " + ", ".join(missing))
    if extra:
        st.error("Unexpected columns: " + ", ".join(extra))
    st.stop()

try:
    X_test = data[features].apply(pd.to_numeric)
    y_test = pd.to_numeric(data["spam"])
except (TypeError, ValueError):
    st.error("All input values and spam labels must be numbers.")
    st.stop()

if not np.isfinite(X_test.to_numpy()).all():
    st.error("Input values must not be blank or infinite.")
    st.stop()

if not y_test.isin([0, 1]).all():
    st.error("The spam column must contain only 0 or 1, with no blanks.")
    st.stop()

y_test = y_test.astype(int)
predictions = model.predict(X_test)
if model_name == "Support Vector Machine":
    scores = model.decision_function(X_test)
else:
    scores = model.predict_proba(X_test)[:, 1]

auc = roc_auc_score(y_test, scores) if y_test.nunique() == 2 else None
results = {
    "Accuracy": accuracy_score(y_test, predictions),
    "AUC": auc,
    "Precision": precision_score(y_test, predictions, zero_division=0),
    "Recall": recall_score(y_test, predictions, zero_division=0),
    "F1": f1_score(y_test, predictions, zero_division=0),
    "MCC": matthews_corrcoef(y_test, predictions),
}

st.subheader(model_name)
st.caption(f"Test rows: {len(data)}")
st.table(pd.DataFrame({
    "Metric": list(results),
    "Score": [f"{value:.4f}" if value is not None else "Not available" for value in results.values()],
}).set_index("Metric"))

if auc is None:
    st.info("AUC needs both 0 and 1 in the spam column.")

st.subheader("Confusion matrix")
matrix = pd.DataFrame(
    confusion_matrix(y_test, predictions, labels=[0, 1]),
    index=["Actual not spam", "Actual spam"],
    columns=["Predicted not spam", "Predicted spam"],
)
st.table(matrix)
