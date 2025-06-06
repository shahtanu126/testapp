import streamlit as st
import cv2
import numpy as np
from PIL import Image
import os
import requests



# Sidebar content
st.sidebar.image('Ahmad Ali.png', use_column_width=True)
st.sidebar.header("**Ahmad Ali Rafique**")
st.sidebar.write("AI & Machine Learning Expert")

st.sidebar.header("About Model")
st.sidebar.info('''This Model is designed for real-time helmet detection using the YOLOv3 (You Only Look Once) model  
    1️⃣ Click on Upload button   
    2️⃣ Upload images to detect helmets  
    3️⃣ See the results ''')
st.sidebar.header("Contact Information")
st.sidebar.write("Feel free to reach out through the following:")
st.sidebar.write("[LinkedIn](https://www.linkedin.com/in/ahmad-ali-rafique/)")
st.sidebar.write("[GitHub](https://github.com/Ahmad-Ali-Rafique/)")
st.sidebar.write("[Email](mailto:arsbussiness786@gmail.com)")
st.sidebar.write("Developed by Ahmad Ali Rafique", unsafe_allow_html=True)

