import streamlit as st
import numpy as np
import cv2
import pandas as pd
import requests
import pytz
import pyperclip
from PIL import Image
from io import BytesIO
from datetime import datetime
from streamlit_cropper import st_cropper

st.set_option("deprecation.showfileUploaderEncoding", False)
st.set_page_config(
    page_title="Fores",
    layout="wide",
    page_icon="💅",
    initial_sidebar_state="expanded",
)

model_product = pd.read_pickle("models/knn-product-smote.pkl")
model_phototype = pd.read_pickle("models/knn-phototype.pkl")

df = pd.read_csv("datasets/foundation/w3ll_people.csv")
df_image = pd.read_csv("datasets/fitzpatrick/fitzpatrick_with_recommendation.csv")
format_file = ["png", "jpg", "jpeg"]

product_label = {
    0: "Dark Neutral (medium dark skin w/ neutral undertones)",
    1: "Fair Golden (fair skin w/ neutral or golden undertones)",
    2: "Fair Pink (fair skin w/ neutral or pink undertones",
    3: "Medium Golden (medium skin w/ golden undertones)",
    4: "Medium Neutral (medium skin w/ neutral undertones)",
    5: "Tan (tan skin w/ neutral undertones)",
}

phototype_label = {0: "I & II", 1: "III", 2: "IV", 3: "V", 4: "VI"}


def query_selected_brand(brand):
    return df if brand == "All brand" else df[df["brand"] == brand]


def process_image(
    img_src,
    realtime_update=True,
    box_color="#0000FF",
    aspect_ratio="1:1",
    degree=0,
):
    col_left, col_right = st.columns(2)
    with col_left:
        img = Image.open(img_src).rotate(degree)

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

        st.image(cropped_image, use_column_width=True)
        st.caption("Cropped Image")

    with col_right:
        cropped_image = np.array(cropped_image)
        hsv_mean = extract_hsv_mean(cropped_image).reshape(1, -1)

        new_data = np.array(hsv_mean)
        h_value = new_data[0][0]
        s_value = new_data[0][1]
        v_value = new_data[0][2]

        features = np.array([h_value, s_value, v_value]).reshape(1, -1)

        st.table(
            pd.DataFrame(
                {
                    "H": [h_value],
                    "S": [s_value],
                    "V": [v_value],
                },
                index=["Features"],
            )
        )

        if st.button("Copy Features"):
            pyperclip.copy(f"{h_value}, {s_value}, {v_value}")
            st.toast("Copied!", icon="✅")

        prediction_product = model_product.predict(features)
        prediction_phototype = model_phototype.predict(features)

        st.markdown(f"#### Phototype:")
        st.markdown(f"###### {phototype_label[prediction_phototype[0]]}")

        st.divider()

        st.markdown(f"#### Recommended Product:")
        st.markdown(f"###### {product_label[prediction_product[0]]}")

        product_index = df[df["imgAlt"] == product_label[prediction_product[0]]].index[
            0
        ]

        product_hex = df["hex"].iloc[product_index]
        st.text(f"Hex: {product_hex}")

        link = df["url"].iloc[product_index]
        link = link.split(",")[0]
        st.link_button(label="Link to Product", url=link)

        # url = df["imgSrc"].iloc[product_index]
        # img = fetch_image(url)
        # st.image(img, channels="BGR", width=60)


def fetch_image(url):
    try:
        response = requests.get(url)
        response.raise_for_status()

        img = Image.open(BytesIO(response.content))

        return img
    except Exception as e:
        print(f"Error: {e}")
        return None


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
    if mode:
        st.toast("Photo taken!", icon="✅")
else:
    mode = st.sidebar.file_uploader(
        label="Upload your photo!",
        type=format_file,
    )
    if mode:
        st.toast("Image uploaded!", icon="✅")

realtime_update = st.sidebar.checkbox(
    label="Update in Real Time",
    value=True,
)

rotation = st.sidebar.slider(
    label="Degree",
    min_value=0,
    max_value=360,
    step=10,
)

with st.sidebar.expander("Crop Utilities"):
    box_color = st.color_picker(
        label="Box Color",
        value="#0EFF00",
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
        degree=rotation,
    )
elif mode is not None:
    process_image(
        img_src=mode,
        realtime_update=realtime_update,
        box_color=box_color,
        aspect_ratio=aspect_ratio,
        degree=rotation,
    )
