import streamlit as st
import numpy as np
from keras.preprocessing import image
import pickle


with open('breastCancer.pkl', 'rb') as file:
    model = pickle.load(file)


def predict(image_path):
    img = image.load_img(image_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    pred = model.predict(img_array)
    return pred


st.title('Breast Cancer Classification')
st.write('Upload an image for breast cancer classification.')

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png"])

if uploaded_file is not None:
    st.image(uploaded_file, caption='Uploaded Image.', use_column_width=True)
    st.write("")
    st.write("Classifying...")

    
    with open('temp_image.png', 'wb') as f:
        f.write(uploaded_file.getbuffer())

    
    pred = predict('temp_image.png')

    if np.argmax(pred) == 0:
        st.write('Prediction: Benign')
    else:
        st.write('Prediction: Malignant')
