import streamlit as st
from PIL import Image
import numpy as np
from tensorflow.keras.models import load_model


model = load_model('model.h5')

# Set the mean and standard deviation used for normalization during training
x_train_mean = 159.77774599641577
x_train_std = 46.49773603445574

# Title of the Streamlit app
st.title("EpidermaCare")

# Upload the image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    try:
        # Open the image file
        image = Image.open(uploaded_file)

        # Display the image
        st.image(image, caption='Uploaded Image.', use_column_width=True)

        # Resize the image to (125, 100)
        image = image.resize((125, 100))

        # Convert the image to a numpy array
        image_array = np.asarray(image)

        # Normalize the image
        image_array = (image_array - x_train_mean) / x_train_std

        # Reshape the image to (1, 100, 125, 3)
        image_array = image_array.reshape(1, 100, 125, 3)

        # Predict
        prediction = model.predict(image_array)

        # Print the prediction
        predicted_class = np.argmax(prediction, axis=1)

        # If you want to map the predicted class index back to the lesion type
        lesion_type_dict = {
            0: 'Melanocytic nevi',
            1: 'Melanoma',
            2: 'Benign keratosis-like lesions ',
            3: 'Basal cell carcinoma',
            4: 'Actinic keratoses',
            5: 'Vascular lesions',
            6: 'Dermatofibroma'
        }

        st.write(f"Predicted class: {predicted_class[0]}")
        st.write(f"Predicted lesion type: {lesion_type_dict[predicted_class[0]]}")

    except Exception as e:
        st.write("Error processing image:", e)
