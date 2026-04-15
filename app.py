import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

st.set_page_config(page_title="Tomato Disease Detection", page_icon="🍅")

@st.cache_resource
def load_model():
    return tf.keras.models.load_model("hybrid_tomato_disease_model.keras")

model = load_model()

class_names = [
    "Bacterial Spot",
    "Early Blight",
    "Late Blight",
    "Leaf Mold",
    "Septoria Leaf Spot",
    "Spider Mites",
    "Target Spot",
    "Yellow Leaf Curl Virus",
    "Mosaic Virus",
    "Healthy"
]

st.title("🍅 Tomato Leaf Disease Detection")

uploaded_file = st.file_uploader(
    "Upload Tomato Leaf Image",
    type=["jpg", "png", "jpeg"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    display_img = image.copy()

    image = image.resize((224, 224))
    img_array = np.array(image, dtype=np.float32) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array, verbose=0)

    pred_idx = np.argmax(prediction[0], axis=1)[0]
    confidence = float(np.max(prediction[0]))
    severity = float(prediction[1][0][0] * 100)

    st.image(display_img, width=300)
    st.success(f"Disease: {class_names[pred_idx]}")
    st.metric("Confidence", f"{confidence:.2%}")
    st.metric("Severity", f"{severity:.2f}%")
