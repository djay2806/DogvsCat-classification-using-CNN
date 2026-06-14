# Project Overview:

This project implements a Convolutional Neural Network (CNN) to classify images as either Dog or Cat. The model is trained on a labeled image dataset and uses deep learning techniques to automatically learn features from images.
To improve model generalization and reduce overfitting, Normalization and Dropout layers are incorporated into the network architecture.

# Model Architecture:

The CNN architecture consists of:

1.Convolutional Layers for feature extraction
2.Max Pooling Layers for dimensionality reduction
3.Normalization Layer to scale pixel values
4.Dropout Layers to reduce overfitting
5.Fully Connected (Dense) Layers for classification
6.Sigmoid Output Layer for binary prediction

# dataset/
│
├── train/
│   ├── cats/
│   └── dogs/
│
├── validation/
│   ├── cats/
│   └── dogs/

# Technologies Used:

Python
TensorFlow / Keras
NumPy
OpenCV / Pillow
Streamlit

# Running the Project:

1.Clone the Repository
Bash
git clone https://github.com/djay2806/dog-cat-classification.git
cd dog-cat-classification

2.Install Dependencies
Bash
pip install -r requirements.txt

3.Run Streamlit App
Bash
streamlit run app.py

# Future Improvements:

1.Data Augmentation
2.Transfer Learning (VGG16, ResNet50, MobileNet)
3.Multi-class Pet Classification
4.Model Explainability using Grad-CAM

# Author:

Dhananjay Patil
GitHub: https://github.com/djay2806
