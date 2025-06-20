import streamlit as st
import numpy as np
import cv2
from PIL import Image
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

# Load model
model = load_model('cnn_model_latest.weights.h5')  # ganti nama file jika perlu

# Label emosi
class_names = ['angry', 'fear', 'happy', 'neutral', 'sad', 'surprise']

# Fungsi preprocessing gambar
def preprocess_image(image):
    image = image.convert('L')  # grayscale
    image = image.resize((48, 48))
    image = img_to_array(image)
    image = image / 255.0
    image = np.expand_dims(image, axis=0)
    return image

# Prediksi dari gambar
def predict_emotion(image):
    processed = preprocess_image(image)
    prediction = model.predict(processed)[0]
    predicted_class = class_names[np.argmax(prediction)]
    confidence = np.max(prediction)
    return predicted_class, confidence

# Title
st.title("Facial Emotion Recognition")

# Pilihan mode input
mode = st.radio("Pilih metode input gambar:", ["📸 Kamera Langsung", "🖼️ Upload Gambar"])

# Kamera langsung
if mode == "📸 Kamera Langsung":
    camera_image = st.camera_input("Ambil gambar wajah")
    if camera_image is not None:
        img = Image.open(camera_image)
        st.image(img, caption="Gambar dari kamera", use_column_width=True)
        label, conf = predict_emotion(img)
        st.success(f"Prediksi Emosi: **{label.upper()}** ({conf:.2%})")

# Upload gambar
elif mode == "🖼️ Upload Gambar":
    uploaded = st.file_uploader("Unggah gambar wajah", type=["jpg", "png", "jpeg"])
    if uploaded is not None:
        img = Image.open(uploaded)
        st.image(img, caption="Gambar yang diunggah", use_column_width=True)
        label, conf = predict_emotion(img)
        st.success(f"Prediksi Emosi: **{label.upper()}** ({conf:.2%})")
