import streamlit as st

st.title("📁 File Upload Example")

# File uploader widget
uploaded_file = st.file_uploader("Upload a file", type=["jpg", "jpeg", "png", "pdf", "txt", "pt"])

# Check if a file is uploaded
if uploaded_file is not None:
    st.success(f"✅ File uploaded: `{uploaded_file.name}`")
else:
    st.info("Please upload a file.")
