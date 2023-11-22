import streamlit as st
import numpy as np
import cv2
import pandas as pd
import requests
import time
from PIL import Image
from io import BytesIO
from streamlit_cropper import st_cropper
from streamlit_drawable_canvas import st_canvas

st.set_option("deprecation.showfileUploaderEncoding", False)

# read the pickle file
model = pd.read_pickle("models/knn_gan_vmean.pkl")
df = pd.read_csv("datasets/foundation/allShades_new.csv")
format_file = ["png", "jpg", "jpeg"]


def process_image(
    img_src,
    realtime_update=True,
    box_color="#0000FF",
    aspect_ratio="1:1",
    drawing_mode="freedraw",
    stroke_width=1,
    stroke_color="#FFFFFF",
):
    col_left, col_right = st.columns(2)

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

        current_time = time.strftime("%b %d, %Y %H:%M:%S")
        st.markdown(f"Photo taken: _{current_time}_")

    with col_right:
        st_canvas(
            background_image=img,
            key="canvas",
            update_streamlit=realtime_update,
            drawing_mode=drawing_mode,
            height=img.height,
            width=img.width,
            stroke_color=stroke_color,
            stroke_width=stroke_width,
        )
        # st.image(img_src, use_column_width=True)
        cropped_image = np.array(cropped_image)
        hsv_mean = extract_hsv_mean(cropped_image).reshape(1, -1)

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
        st.text(f"Hex: {hex_code}")

        desc = df["imgAlt"][product_index]
        st.text(f"Description: {desc}")

        link = df["url"][product_index]
        link = link.split(",")[0]
        st.markdown(f"Link to [Product]({link})")

        url = df["imgSrc"][product_index]
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


def query_selected_brand(brand):
    return df[df["brand"] == brand]


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

with st.sidebar.expander("Draw Utilities"):
    stroke_width = st.slider(
        label="Stroke Width",
        min_value=1,
        max_value=10,
        value=1,
        step=1,
    )
    stroke_color = st.color_picker(
        label="Stroke Color",
        value="#000000",
    )
    drawing_mode = st.selectbox(
        label="Drawing Tool",
        options=["freedraw", "line", "rect", "circle", "transform"],
    )

st.sidebar.markdown(
    "Made by [Janice Claresta Lingga](https://github.com/janeclrst) 🐈",
)

if options == "Camera" and mode is not None:
    process_image(
        img_src=mode,
        realtime_update=realtime_update,
        box_color=box_color,
        aspect_ratio=aspect_ratio,
        drawing_mode=drawing_mode,
        stroke_width=stroke_width,
        stroke_color=stroke_color,
    )
elif mode is not None:
    process_image(
        img_src=mode,
        realtime_update=realtime_update,
        box_color=box_color,
        aspect_ratio=aspect_ratio,
        drawing_mode=drawing_mode,
        stroke_width=stroke_width,
        stroke_color=stroke_color,
    )
