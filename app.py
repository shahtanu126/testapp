import streamlit as st
from PIL import Image
import torch
import tempfile
import os

# Load YOLO model
@st.cache_resource
def load_model(model_path='yolov5s.pt'):
    model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path)
    return model

# Inference function
def run_inference(model, image):
    results = model(image)
    return results

# Streamlit UI
st.title("🔍 YOLO Model Checker")
st.write("Upload an image to test your YOLO object detection model.")

# Sidebar model upload
model_file = st.sidebar.file_uploader("Upload YOLO Model (.pt)", type=['pt'])

if model_file is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pt") as tmp:
        tmp.write(model_file.read())
        model_path = tmp.name
    model = load_model(model_path)

    # Image uploader
    uploaded_image = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])
    if uploaded_image is not None:
        image = Image.open(uploaded_image)
        st.image(image, caption="Uploaded Image", use_column_width=True)

        if st.button("Run Detection"):
            with st.spinner("Running YOLO detection..."):
                results = run_inference(model, image)
                st.image(results.render()[0], caption="Detection Output", use_column_width=True)
                st.success("Detection complete!")

else:
    st.info("Please upload a YOLO model (.pt file) from the sidebar.")
