import pickle
import streamlit as st
import tensorflow as tf
import numpy as np
from sklearn.svm import SVC

pickle_in = open('SVMClassifier.pkl', 'rb')
SVM = pickle.load(pickle_in)

st.title("MNIST Digit Recogniser")
st.header("By Karan Kumar Singh")

image2 = st.file_uploader(label="Upload image", type=['jpg', 'png'])

if st.button("Upload Image"):
    if image2 is not None:
        # Manual Image processing
        resized_image = tf.keras.preprocessing.image.load_img(image2, target_size=(28,28))  # Resizing it to 28x28
        resized_image = np.array(resized_image)  # Converting it to Array for further processes
        resized_image = resized_image / 255.  # Normalising
        tensorImg = tf.image.rgb_to_grayscale(resized_image)  # Convert Grayscale(28,28,1) instead of RGB(28,28,3)
        tensorImg = tf.squeeze(tensorImg, axis=-1)  # shape is (28, 28)
        tensorImg = 1.0 - tensorImg  # Altering the colour Black -> White & White -> Black to make background Black
        imagearr = np.array(tensorImg)  # Converting into np.array bcz now it is tensor (TensorFlow Object)
        finalimg = imagearr.reshape(1, 784)  # Converting 28 x 28 to 1 x 784 2D Array

        # Filtering out unnecessary pixels by converting it to 0 or 1 to increase accuracy
        for i in range(784):
            if finalimg[0][i] <= 0.39:
                finalimg[0][i] = 0.0
            elif finalimg[0][i] >= 0.66:
                finalimg[0][i] = 1.0

        st.success(f"Predicted Number: {SVM.predict(finalimg)[0]}")
    else:
        st.write("Make sure you image is in JPG/PNG Format.")
