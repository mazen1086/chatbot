import streamlit as st
import requests

# Streamlit interface for user interaction
st.title("Churn Prediction Chatbot")
user_input = st.text_area("Enter customer details:")


if st.button("Predict and Generate Response"):
    try:

        response = requests.post("http://192.168.126.13:5000/api/v1/response", json={"user_input": user_input}).json().get("response")

        st.success("Generated Response:")
        st.write(response)
    except Exception as e:
        st.error(f"Error: {e}")
