import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np

st.set_page_config(page_title="Face Mask Detector")


@st.cache_resource
def load_my_model():
    return tf.keras.models.load_model('model.h5')

model = load_my_model()

st.title("😷 Face Mask Detection System")
st.write("Upload a photo to see if the person is wearing a mask.")


file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if file is not None:
    
    image = Image.open(file).convert('RGB')
    st.image(image, caption='Uploaded Image', use_column_width=True)
    
    
    img_array = np.array(image.resize((224, 224))) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    
    prediction = model.predict(img_array)
    
    
    if prediction[0][0] < 0.5:
        st.success("✅ Prediction: Mask Detected!")
    else:
        st.error("🚨 Prediction: No Mask Detected")
