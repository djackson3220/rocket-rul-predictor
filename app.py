import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="Engine Life Predictor", layout="wide")
st.title("Rocket Engine Remaining Useful Life Predictor")
st.write("Upload your sensor data file and get a prediction of the engine's Remaining Useful Life (RUL).")

uploaded_file = st.file_uploader("Choose a CSV file with engine sensor readings", type=["csv"])

model = None
try:
    model = joblib.load("model.pkl")
except Exception:
    st.warning("Model file not found. Please add 'model.pkl' to the repo.")

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.subheader("Uploaded Data Preview")
    st.dataframe(df.head())

    if model:
        predictions = model.predict(df)
        df['Predicted RUL'] = predictions
        st.subheader("Predicted Remaining Useful Life")
        st.line_chart(df['Predicted RUL'])
    else:
        st.error("Cannot predict without a loaded model.")

st.sidebar.header("Repository Setup")
st.sidebar.markdown(
    "1. Fork this repo on GitHub.\n"
    "2. Add your trained model as `model.pkl` in the project root.\n"
    "3. Update `requirements.txt` with dependencies (`streamlit`, `pandas`, `scikit-learn`, `joblib`).\n"
    "4. Deploy on Streamlit Cloud or your preferred hosting."
)
