import streamlit as st
import numpy as np
import cv2
import pandas as pd
import requests
import pytz
from PIL import Image
from io import BytesIO
from datetime import datetime
from streamlit_cropper import st_cropper

st.set_option("deprecation.showfileUploaderEncoding", False)

model_product = pd.read_pickle("models/knn_fitzpatrick_vmean_product.pkl")
model_phototype = pd.read_pickle("models/knn_fitzpatrick_vmean_phototype.pkl")

df = pd.read_csv("datasets/foundation/w3ll_people.csv")
df_image = pd.read_csv("datasets/fitzpatrick/fitzpatrick_with_recommendation.csv")
format_file = ["png", "jpg", "jpeg"]


def query_selected_brand(brand):
    return df if brand == "All brand" else df[df["brand"] == brand]


def process_image(
    img_src,
    realtime_update=True,
    box_color="#0000FF",
    aspect_ratio="1:1",
):
    col_left, col_right = st.columns(2)

    model_product.fit(
        df_image["Value"].values.reshape(-1, 1),
        df_image["product"].values.reshape(-1, 1),
    )
    model_phototype.fit(
        df_image["Value"].values.reshape(-1, 1),
        df_image["phototype"].values.reshape(-1, 1),
    )

    with col_left:
        img = Image.open(img_src)

        if not realtime_update:
            st.write("Double tap on the image to save crop")

        cropped_image = st_cropper(
            img,
            key="cropper",
            realtime_update=realtime_update,
            box_color=box_color,
            aspect_ratio=aspect_ratio,
            stroke_width=4,
        )

        id_tz = pytz.timezone("Asia/Jakarta")
        current_time = datetime.now(id_tz).strftime("%b %d, %Y %H:%M:%S")
        st.markdown(f"Photo taken: _{current_time}_")

    with col_right:
        st.image(img_src, use_column_width=True)
        cropped_image = np.array(cropped_image)
        hsv_mean = extract_hsv_mean(cropped_image).reshape(1, -1)

        new_data = np.array(hsv_mean)
        h_value = new_data[0][0]
        s_value = new_data[0][1]
        v_value = new_data[0][2]

        features = np.array([h_value, s_value, v_value]).reshape(1, -1).T
        st.text(f"Features: {features}")

        prediction_product = model_product.predict(features)
        prediction_phototype = model_phototype.predict(features)

        st.text(f"Phototype prediction: {prediction_phototype[0]}")
        st.text(f"Product prediction: {prediction_product[0]}")

        product_index = df[df["imgAlt"] == prediction_product[0]].index[0]

        product_hex = df["hex"].iloc[product_index]
        st.text(f"Hex: {product_hex}")

        link = df["url"].iloc[product_index]
        link = link.split(",")[0]
        st.markdown(f"Link to [Product]({link})")

        url = df["imgSrc"].iloc[product_index]
        img = fetch_image(url)
        st.image(img, channels="BGR", width=60)


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
    cropped_image = img[y1:y2, x1:x2]
    return cv2.resize(cropped_image, None, fx=zoom_factor, fy=zoom_factor)


def extract_hsv_mean(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h = np.mean(hsv[:, :, 0])
    s = np.mean(hsv[:, :, 1])
    v = np.mean(hsv[:, :, 2])
    return np.array([h, s, v])


def convert_to_percentage(value):
    return value / 255


st.title("Fores (Foundation Recommender System)")
st.subheader("AI Powered Foundation Recommender System Using")

options = st.sidebar.radio(
    label="Mode",
    horizontal=True,
    options=["Camera", "Upload"],
    index=0,
)


if options == "Camera":
    mode = st.sidebar.camera_input(label="Take a photo")
else:
    mode = st.sidebar.file_uploader(
        label="Upload your photo!",
        type=format_file,
    )

unique_brands = df["brand"].unique().tolist()
unique_brands.insert(0, "All brand")

realtime_update = st.sidebar.checkbox(
    label="Update in Real Time",
    value=True,
)

with st.sidebar.expander("Crop Utilities"):
    box_color = st.color_picker(
        label="Box Color",
        value="#0000FF",
    )
    aspect_choice = st.radio(
        label="Aspect Ratio",
        options=["1:1", "16:9", "4:3", "2:3", "Free"],
        index=0,
    )
    aspect_dict = {
        "1:1": (1, 1),
        "16:9": (16, 9),
        "4:3": (4, 3),
        "2:3": (2, 3),
        "Free": None,
    }
    aspect_ratio = aspect_dict[aspect_choice]

st.sidebar.markdown(
    "Made by [Janice Claresta Lingga](https://github.com/janeclrst) 🐈",
)

if options == "Camera" and mode is not None:
    process_image(
        img_src=mode,
        realtime_update=realtime_update,
        box_color=box_color,
        aspect_ratio=aspect_ratio,
    )
elif mode is not None:
    process_image(
        img_src=mode,
        realtime_update=realtime_update,
        box_color=box_color,
        aspect_ratio=aspect_ratio,
    )
