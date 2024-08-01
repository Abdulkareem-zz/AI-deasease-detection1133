import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import tempfile

# Load the model
model = load_model('disease_detection_model.h5')

# Streamlit app
st.title("AI-Powered Early Disease Detection")

uploaded_file = st.file_uploader("Upload a medical image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Save the uploaded file to a temporary file
    with tempfile.NamedTemporaryFile(delete=False) as temp_file:
        temp_file.write(uploaded_file.read())
        temp_file_path = temp_file.name

    img = image.load_img(temp_file_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0

    prediction = model.predict(img_array)
    result = int(np.round(prediction[0][0]))

    st.image(img, caption='Uploaded Image.', use_column_width=True)
    st.write(f'Prediction: {result}')
