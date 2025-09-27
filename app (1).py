import streamlit as st
import tensorflow as tf
from io import BytesIO
import PIL.Image as Image

# Categories and model
CATEGORIES = ['Normal', 'Pneumonia']
img_size = 100
model = tf.keras.models.load_model('pre_trained_model_2.h5')
print(' Model Loaded')

# ---- Page Config ----
st.set_page_config(page_title="Pneumonia Detection", layout="centered")

# ---- Custom CSS ----
st.markdown("""
    <style>
        body {
            font-family: 'Poppins', sans-serif;
        }
        .main {
            background-color: #f9f9f9;
            padding: 2rem;
            border-radius: 12px;
        }
        .stButton>button {
            background-color: #0d47a1;
            color: white;
            border-radius: 8px;
            padding: 0.6rem 1.2rem;
            border: none;
            font-size: 16px;
            font-weight: 500;
        }
        .stButton>button:hover {
            background-color: #1565c0;
        }
        .upload-box {
            border: 2px dashed #0d47a1;
            padding: 2rem;
            border-radius: 12px;
            text-align: center;
            background-color: white;
        }
    </style>
""", unsafe_allow_html=True)

# ---- Title ----
st.title(" Pneumonia Detection ")
st.write("Upload your chest X-ray and let our model assist in detecting Pneumonia.")

# ---- Upload Section ----
file = st.file_uploader("Upload a Chest X-ray", type=["jpeg", "jpg", "png"])

if file is not None:
    # Show uploaded image
    st.image(file, caption="Uploaded X-ray", use_column_width=True)

    # Preprocess
    img = Image.open(BytesIO(file.read())).convert('RGB')
    img = img.resize((img_size, img_size))
    new_array = tf.keras.preprocessing.image.img_to_array(img)
    new_array = new_array.reshape(-1, img_size, img_size, 3)

    # Prediction button
    if st.button(" Predict"):
        prediction = model.predict(new_array / 255.0)
        confidence = prediction[0][0]

        predicted_class = CATEGORIES[int(round(confidence))]
        final_confidence = confidence if confidence > 0.5 else 1 - confidence

        st.success(f"### Result: **{predicted_class}**")
        st.info(f"Confidence: **{final_confidence * 100:.2f}%**")

