import tensorflow as tf
import streamlit as st
from PIL import Image
import numpy as np
import os
from huggingface_hub import hf_hub_download

model_path = "dog_cat_model.h5"

if not os.path.exists(model_path):
    with st.spinner("Downloading model from Hugging Face..."):
        # This downloads the file from your public Hugging Face model repo
        model_path = hf_hub_download(
            repo_id="johnsnow321/CatvsDog-Classification", 
            filename="dog_cat_model.h5"
        )
        st.success("Download complete!")

model = tf.keras.models.load_model(model_path)


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
