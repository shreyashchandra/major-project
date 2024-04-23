import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import pickle

# Load the model
with open('breastCancer.pkl', 'rb') as file:
    model = pickle.load(file)

def predict(image_path):
    # Load and preprocess the image
    img = image.load_img(image_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    
    # Predict the class probabilities
    pred = model.predict(img_array)
    return pred

def main():
    st.title('Breast Cancer Classification')
    st.write('Upload an image for breast cancer classification.')

    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png"])

    if uploaded_file is not None:
        # Display the uploaded image
        st.image(uploaded_file, caption='Uploaded Image.', use_column_width=True)
        st.write("")
        st.write("Classifying...")
        
        try:
            # Save the uploaded image temporarily
            with open('temp_image.png', 'wb') as f:
                f.write(uploaded_file.read())

            # Predict the class
            pred = predict('temp_image.png')

            if np.argmax(pred) == 0:
                st.write('Prediction: Benign')
            else:
                st.write('Prediction: Malignant')
        
        except Exception as e:
            st.error("An error occurred during classification: {}".format(e))
        finally:
            # Delete the temporary file
            if os.path.exists('temp_image.png'):
                os.remove('temp_image.png')

if __name__ == "__main__":
    main()
