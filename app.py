import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import json
import gdown
import os

# ---------------------------
# تحميل الموديل من Google Drive
# ---------------------------
MODEL_PATH = "plant_disease_prediction_model.keras"
if not os.path.exists(MODEL_PATH):
    file_id = "1dBxiCkGL17RS1P5qsOhtRGezReiXMZyg"
    url = f"https://drive.google.com/uc?id={file_id}"
    gdown.download(url, MODEL_PATH, quiet=False)

# ---------------------------
# تحميل الموديل
# ---------------------------
model = load_model(MODEL_PATH)

# ---------------------------
# تحميل أسماء الفئات من JSON
# ---------------------------
with open("class_indices.json", "r") as f:
    class_indices = json.load(f)
classes = [name for name, idx in sorted(class_indices.items(), key=lambda x: x[1])]

# ---------------------------
# واجهة Streamlit
# ---------------------------
st.set_page_config(page_title="Plant Disease Prediction 🌱", layout="centered")
st.title("Plant Disease Prediction 🌱")

uploaded_file = st.file_uploader("Upload an image of the plant leaf", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file)
    st.image(img, caption='Uploaded Leaf', use_column_width=True)

    img = img.resize((224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0

    prediction = model.predict(img_array)
    class_idx = np.argmax(prediction, axis=1)[0]
    st.success(f"Prediction: {classes[class_idx]}")
