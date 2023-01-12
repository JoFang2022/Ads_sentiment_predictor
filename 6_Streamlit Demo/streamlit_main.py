import streamlit as st
import requests

st.title("Welcome")
st.markdown("Please upload file here: ")
st.sidebar.markdown("Here you can find out more information about your video")
f = st.file_uploader("Upload file")

if st.button("Predict"):
    res = requests.post("http://0.0.0.0:8000/", files=dict(file=f))
    st.subheader(res.text)


# streamlit run streamlit_main.py
