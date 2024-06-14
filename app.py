import tensorflow as tf 
from tensorflow import keras
from tensorflow.keras.models import load_model
import streamlit as st 
import numpy as np

#Load the model
model = load_model(r'C:\Users\ABHI\AppData\Local\Packages\PythonSoftwareFoundation.Python.3.9_qbz5n2kfra8p0\LocalCache\local-packages\Python39\Scripts\image_class_model\image_classification_test_2.keras')

data_cat = [
    'D_fruit', 'apple', 'audi', 'bmw', 'cat',
    'dog', 'grape', 'horse', 'jeep', 'lion'
]

#Path 
img_path = r'C:\Users\ABHI\AppData\Local\Packages\PythonSoftwareFoundation.Python.3.9_qbz5n2kfra8p0\LocalCache\local-packages\Python39\Scripts\image_class_model\grape.jpg'


img_width, img_height = 180, 180


image = tf.keras.utils.load_img(img_path, target_size=(img_width, img_height))
img_arr = tf.keras.utils.img_to_array(image)
img_bat = tf.expand_dims(img_arr, 0)


predict = model.predict(img_bat)
score = tf.nn.softmax(predict)

# Display in Streamlit
st.title("Image Classification")
st.write("This app classifies images into different categories using a pre-trained neural network model.")
st.image(image, caption='Input Image', use_column_width=True)
st.write(f'The given image is classified as {data_cat[np.argmax(score)]}')

# Add a sidebar (we can remove this)
st.sidebar.title("Image Classification App")
st.sidebar.write("This sidebar contains additional information and settings for the image classification app.")
st.sidebar.image(image, caption='Input Image', use_column_width=True)
st.sidebar.write(f'Predicted Class: **{data_cat[np.argmax(score)]}**')
#st.sidebar.write(f'Accuracy: **{np.max(score) * 100:.2f}%**')

# Add more sections
st.subheader("Model Information")
st.write("""
    This model is a Convolutional Neural Network (CNN) trained on a dataset of various images to classify them into one of the following categories:
    - D_fruit
    - Apple
    - Audi
    - BMW
    - Cat
    - Dog
    - Grape
    - Horse
    - Jeep
    - Lion
""")

st.subheader("How it works")
st.write("""
    1. The image is loaded and resized to the target size.
    2. The image is then converted to an array and expanded to match the input shape required by the model.
    3. The model predicts the class of the image and returns the probabilities.
    4. The class with the highest probability is selected as the predicted class.
""")

st.subheader("Try it yourself!")
st.write("Upload an image to see the classification result.")
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png"])
if uploaded_file is not None:
    # Load and preprocess the uploaded image
    uploaded_image = tf.keras.utils.load_img(uploaded_file, target_size=(img_width, img_height))
    uploaded_img_arr = tf.keras.utils.img_to_array(uploaded_image)
    uploaded_img_bat = tf.expand_dims(uploaded_img_arr, 0)
    
    # Predict the class of the uploaded image
    uploaded_predict = model.predict(uploaded_img_bat)
    uploaded_score = tf.nn.softmax(uploaded_predict)
    
    # Display the uploaded image and prediction
    st.image(uploaded_image, caption='Uploaded Image', use_column_width=True)
    st.write(f'The given image is classified as {data_cat[np.argmax(uploaded_score)]}')


