import streamlit as st
import cv2
import numpy as np
from PIL import Image
import os
import requests

# Function to download files
def download_file(url, local_filename):
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        with open(local_filename, 'wb') as f:
            for chunk in r.iter_content(chunk_size=8192): 
                f.write(chunk)

# URLs of the files hosted externally
file_urls = {
    "yolov3.weights": "https://pjreddie.com/media/files/yolov3.weights",
    "yolov3.cfg": "https://github.com/pjreddie/darknet/raw/master/cfg/yolov3.cfg",
    "coco.names": "https://github.com/pjreddie/darknet/raw/master/data/coco.names"
}

# Download the files if not present
for filename, url in file_urls.items():
    if not os.path.exists(filename):
        st.info(f"Downloading {filename}...")
        download_file(url, filename)
        st.success(f"Downloaded {filename}!")

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

