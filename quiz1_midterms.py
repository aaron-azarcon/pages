import streamlit as st
import numpy as np
import cv2
import matplotlib.pyplot as plt


def orig(picture):
    height, width = picture.shape[:2]

    return orig


def translation_1(picture, bx, by, tx, ty):
    height, width = picture.shape[:2]
    
    bx = 0
    by = 0
    nx = bx + tx
    ny = by + ty

    m_translation_ = np.float32([[1,0,nx],
                                 [0,1,ny]])
    translated_img_ = cv2.warpAffine(picture, m_translation_, dsize=(width,height))
    return translated_img_

def main():
    st.title("Image Translate")

    uploaded_file = st.file_uploader("Choose an image file", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = cv2.imdecode(np.fromstring(uploaded_file.read(), np.uint8), 1)
        st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), caption="Original Image", use_column_width=True)

    # picture = ["pic1.jpg", "pic2.jpg", "pic3.jpg", "pic4.jpeg", "pic5.png"]
    bx = int
    by = int
    tx = st.slider("How much would you like x to move?:")
    ty = st.slider("How much would you like y to move?:")

    translated_image = translation_1(image, bx, by, tx, ty)
    st.image(cv2.cvtColor(translated_image, cv2.COLOR_BGR2RGB),caption="Image Translation", use_column_width=True)

main()