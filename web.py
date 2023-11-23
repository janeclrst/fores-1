import streamlit as st
import numpy as np
import cv2
import pandas as pd
import requests
import pytz
import time
from PIL import Image
from io import BytesIO
from datetime import datetime
from streamlit_cropper import st_cropper
from streamlit_drawable_canvas import st_canvas

st.set_option("deprecation.showfileUploaderEncoding", False)

# read the pickle file
model = pd.read_pickle("models/knn_gan_vmean.pkl")
df = pd.read_csv("datasets/foundation/allShades_new.csv")
format_file = ["png", "jpg", "jpeg"]


def query_selected_brand(brand):
    return df if brand == "All brand" else df[df["brand"] == brand]


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

    filtered_df = query_selected_brand(selected_brand)

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

        v_data = filtered_df.get("Value")
        calc = np.array([np.round(np.abs(v_percentage - v_data), 2)])
        nearest_value = np.array([np.min(calc)])

        indices = np.array(np.where(calc == nearest_value)[1])

        if indices.size > 0:
            brand_index = indices[0]
            brand = filtered_df["brand"].iloc[brand_index]
            st.text(f"Brand: {brand}")

            product_index = indices[0]

            st.text(f"Product: {filtered_df['product'].iloc[product_index]}")

            hex_code = filtered_df["hex"].iloc[product_index]
            st.text(f"Hex: {hex_code}")

            desc = filtered_df["imgAlt"].iloc[product_index]
            st.text(f"Description: {desc}")

            link = filtered_df["url"].iloc[product_index]
            link = link.split(",")[0]
            st.markdown(f"Link to [Product]({link})")

            url = filtered_df["imgSrc"].iloc[product_index]
            img = fetch_image(url)
            st.image(img, channels="BGR", width=60)
        else:
            st.text("Brand or product not found")


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

selected_brand = st.selectbox(
    label="Select Brand",
    options=unique_brands,
    index=0,
    key="Brand Selectbox",
    disabled=mode is None,
    help="Select a brand to filter the products",
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
