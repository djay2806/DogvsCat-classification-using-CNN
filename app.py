import tensorflow as tf
import streamlit as st
from PIL import Image
import numpy as np

model = tf.keras.models.load_model("dog_cat_model.h5")

st.title("Dog vs Cat Classifier")

uploaded_file = st.file_uploader(
    "Upload an Image",
    type=["jpg","jpeg","png"]

)

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image,caption="Uploaded Image",width=300)
    image = image.convert('RGB')
    image=image.resize((256,256))
    image_array=np.array(image)

    image_array = image_array/255.0
    image_array=image_array.reshape((1,256,256,3))


    prediction = model.predict(image_array)
    
    if prediction[0][0] > 0.5:
      st.write("Dog") # This might be backwards!
    else:
      st.write("Cat")
