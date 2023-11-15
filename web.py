import streamlit as st
import numpy as np
import cv2
import pandas as pd
from io import StringIO

# read the pickle file
model = pd.read_pickle("models/knn_gan_vmean.pkl")
df = pd.read_csv("datasets/foundation/maybelline_new.csv")


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


def convert_to_percentage(value):
    return value / 255


uploaded_file = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    opencv_img = cv2.imdecode(file_bytes, 1)

    zoom_image = zoom_center(opencv_img, 5)
    hsv_mean = extract_hsv_mean(zoom_image).reshape(1, -1)

    new_data = np.array(hsv_mean)
    v_value = new_data[0][2]
    st.text(f"V value: {v_value}")
    prediction = model.predict(v_value.reshape(1, -1))
    st.text(f"Prediction: {prediction[0]}")
    v_percentage = convert_to_percentage(v_value).round(2)
    st.text(f"V percentage: {v_percentage}")
    v_data = df.get("V")
    calc = np.array([np.round(np.abs(v_data - v_percentage), 2)])
    nearest_data = np.min(calc)
    st.text(f"Nearest data: {nearest_data}")
    # show the product using st.text
    product_index = df.iloc[np.where(calc == nearest_data)[1][0], 0]
    st.text(df["product"][product_index])
