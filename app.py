# streamlit_app.py
import streamlit as st
import os
import pandas as pd
import cv2
from datetime import datetime

st.set_page_config(layout="wide")
st.title("Helmet & Number Plate Detection Dashboard")

date_folder = st.sidebar.text_input("Log Folder Date", datetime.now().strftime("%d-%m-%y"))
excel_file = os.path.join(date_folder, f"{date_folder}.xlsx")

if os.path.exists(excel_file):
    df = pd.read_excel(excel_file)
    st.success(f"Loaded log: {excel_file}")
    st.dataframe(df.tail(20))
    st.download_button("Download Excel", open(excel_file, "rb"), file_name=f"{date_folder}.xlsx")
else:
    st.warning("No log file found for this date.")

# Show image previews
img_dir = date_folder
if os.path.exists(img_dir):
    img_files = sorted([f for f in os.listdir(img_dir) if f.endswith(".jpg")])
    st.subheader("Recently Captured Plates")
    cols = st.columns(5)
    for i, img in enumerate(img_files[-10:]):
        img_path = os.path.join(img_dir, img)
        cols[i % 5].image(img_path, caption=img, width=150)
else:
    st.info("No image directory found yet.")
