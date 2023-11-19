import streamlit as st
import numpy as np
import cv2
import pandas as pd
import requests
from PIL import Image
from io import BytesIO

# read the pickle file
model = pd.read_pickle("models/knn_gan_vmean.pkl")
df = pd.read_csv("datasets/foundation/allShades_new.csv")


def fetch_image(url):
    try:
        response = requests.get(url)
        response.raise_for_status()

        img = Image.open(BytesIO(response.content))

        return img
    except Exception as e:
        print(f"Error: {e}")
        return None


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


st.title("Fores (Foundation Recommender System)")

# uploaded_file = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])
cam = st.camera_input(label="Take a photo")

# if uploaded_file is not None:
if cam is not None:
    col1, col2 = st.columns(2)

    # file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    file_bytes = cam.getvalue()
    # opencv_img = cv2.imdecode(file_bytes, 1)
    opencv_img = cv2.imdecode(np.frombuffer(file_bytes, np.uint8), cv2.IMREAD_COLOR)

    # zoom_ratio = st.slider("Zoom ratio", min_value=1, max_value=10, step=1)

    with col1:
        zoom_image = zoom_center(opencv_img, zoom_factor=zoom_ratio)
        hsv_mean = extract_hsv_mean(zoom_image).reshape(1, -1)

        new_data = np.array(hsv_mean)
        v_value = new_data[0][2]
        prediction = model.predict(v_value.reshape(1, -1))
        st.text(f"Phototype prediction: {prediction[0]}")

        v_percentage = convert_to_percentage(v_value).round(2)

        v_data = df.get("Value")
        calc = np.array([np.round(np.abs(v_percentage - v_data), 2)])
        nearest_value = np.array([np.min(calc)])

        brand = df["brand"][np.array(np.where(calc == nearest_value)[1][0])]
        st.text(f"Brand: {brand}")

        product_index = np.array(np.where(calc == nearest_value)[1][0])
        st.text(f"Product: {df['product'][product_index]}")

        hex_code = df["hex"][product_index]
        st.text(f"Hex: {df['hex'][product_index]}")

        desc = df["imgAlt"][product_index]
        st.text(f"Description: {desc}")

        link = df["url"][product_index]
        link = link.split(",")[0]
        st.markdown(f"Link to [Product]({link})")

        url = df["imgSrc"][product_index]
        img = fetch_image(url)
        st.image(img, channels="BGR", width=60)

    with col2:
        st.image(zoom_image, channels="BGR", width=300)
