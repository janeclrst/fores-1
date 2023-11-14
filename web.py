import streamlit as st
import pandas as pd
import numpy as np
import cv2

# read the pickle file
model = pd.read_pickle("models/knn_gan_vmean.pkl")


def zoom_center(img, zoom_factor=2):
    x_width = img.shape[1]
    y_height = img.shape[0]

    x1 = int(0.5 * x_width * (1 - 1 / zoom_factor))
    y1 = int(0.5 * y_height * (1 - 1 / zoom_factor))

    x2 = int(x_width - 0.5 * x_width * (1 - 1 / zoom_factor))
    y2 = int(y_height - 0.5 * y_height * (1 - 1 / zoom_factor))

    # Crop then scale
    cropped_img = img[y1:y2, x1:x2]
    return cv2.resize(cropped_img, None, fx=zoom_factor, fy=zoom_factor)


def extract_hsv_mean(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h = np.mean(hsv[:, :, 0])
    s = np.mean(hsv[:, :, 1])
    v = np.mean(hsv[:, :, 2])
    return np.array([h, s, v])


uploaded_file = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    # display the image
    # st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)

    # zoom the imported image
    # img = cv2.imread(uploaded_file.name)
    # img = zoom_center(img, 2)
    st.text(uploaded_file)
    st.image(uploaded_file, caption="Zoomed Image", use_column_width=True)
