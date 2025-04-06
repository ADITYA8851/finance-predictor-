
import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt

st.title("Stock Price Predictor with Visualization")

uploaded_file = st.file_uploader("Upload stock data CSV with features", type="csv")
if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    st.write("Uploaded Data", data.head())

    model = joblib.load("stock_model.pkl")
    predictions = model.predict(data)
    data['Predicted Price'] = predictions

    st.write("Predicted Stock Prices", data)

    st.subheader("Prediction Graph")
    fig, ax = plt.subplots()
    ax.plot(data.index, data['Predicted Price'], label='Predicted Price', color='blue', marker='o')
    ax.set_xlabel("Index")
    ax.set_ylabel("Price")
    ax.set_title("Predicted Stock Prices Over Time")
    ax.legend()
    st.pyplot(fig)
