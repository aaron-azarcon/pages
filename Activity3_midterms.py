import numpy as np
import cv2
import matplotlib.pyplot as plt
import streamlit as st

def translate_image(image, dx, dy):
    translation_matrix = np.float32([[1, 0, dx], [0, 1, dy]])
    translated_image = cv2.warpAffine(image, translation_matrix, (image.shape[1], image.shape[0]))
    return translated_image

def rotate_image():
    rotation_matrix = cv2.getRotationMatrix2D((width/2, height/2), angle, 1)
    rotated_image = cv2.warpAffine(image, rotation_matrix, (width, height))
    return rotated_image

def scale_image():
    fx = float(input("Enter the scaling factor for X axis: "))
    fy = float(input("Enter the scaling factor for Y axis: "))
    scaled_image = cv2.resize(image, None, fx=fx, fy=fy, interpolation=cv2.INTER_LINEAR)
    return scaled_image

def reflect_image():
    if flip_direction == '1':
        flipped_image = cv2.flip(image, 1)
    elif flip_direction == '2':
        flipped_image = cv2.flip(image, 0)
    elif flip_direction == '3':
        flipped_image = cv2.flip(image, -1)
    return flipped_image

def shear_image():
    sx = float(input("Enter the shear factor along the X axis: "))
    sy = float(input("Enter the shear factor along the Y axis: "))
    shear_matrix = np.float32([[1, sx, 0], [sy, 1, 0]])
    sheared_image = cv2.warpAffine(image, shear_matrix, (image.shape[1], image.shape[0]))

def main():
    # for translation
    st.title("Image Translate")
    
    uploaded_file = st.file_uploader("Choose an image file", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = cv2.imdecode(np.fromstring(uploaded_file.read(), np.uint8), 1)
        st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), caption="Original Image", use_column_width=True)

        dx = st.slider("Enter the amount to translate along the X axis", -100, 100, 0)
        dy = st.slider("Enter the amount to translate along the Y axis", -100, 100, 0)
        if st.button("Translate"):
            translated_image = translate_image(image, dx, dy)
            st.image(cv2.cvtColor(translated_image, cv2.COLOR_BGR2RGB),caption="Image Translation", use_column_width=True)

    # For rotation
    st.title("Image Rotate")
    uploaded_file = st.file_uploader("Choose an image file", type=["jpg", "jpeg", "png"], key="rotation_key")
    
    if uploaded_file is not None:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, 1)
        angle = st.slider("Enter the rotation angle in degrees", -180, 180, 0)
        height, width = image.shape[:2]
        rotation_matrix = cv2.getRotationMatrix2D((width/2, height/2), angle, 1)
        rotated_image = cv2.warpAffine(image, rotation_matrix, (width, height))
        st.image(cv2.cvtColor(rotated_image, cv2.COLOR_BGR2RGB), caption="Image Rotation",use_column_width=True)

    # for scaling
    st.title("Image Scale")
    uploaded_file = st.file_uploader("Choose an image file", type=["jpg", "jpeg", "png"], key="scale_key")
    if uploaded_file is not None:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), caption="Original Image", use_column_width=True)

        st.write("Please enter a number between decimals (0.05 ~ 1.5)")
        fx = st.slider("Enter the scaling factor for X axis", 0.05, 1.5, 1.0, 0.05)
        fy = st.slider("Enter the scaling factor for Y axis", 0.05, 1.5, 1.0, 0.05)

        scaled_image = cv2.resize(image, None, fx=fx, fy=fy, interpolation=cv2.INTER_LINEAR)
        st.image(cv2.cvtColor(scaled_image, cv2.COLOR_BGR2RGB), caption="Scaled Image", use_column_width=True)

    # for reflection
    st.title("Image Reflection")
    uploaded_file = st.file_uploader("Choose an image file", type=["jpg", "jpeg", "png"], key="reflection_key")
    if uploaded_file is not None:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), caption="Original Image", use_column_width=True)

        flip_direction = st.selectbox("Enter flip direction", ('horizontal', 'vertical', 'both'))
        if flip_direction == 'horizontal':
            flipped_image = cv2.flip(image, 1)
        elif flip_direction == 'vertical':
            flipped_image = cv2.flip(image, 0)
        elif flip_direction == 'both':
            flipped_image = cv2.flip(image, -1)
        else:
            st.error("Invalid direction entered")
            return
        st.image(cv2.cvtColor(flipped_image, cv2.COLOR_BGR2RGB), caption="Image Reflection")

    # forr shearing
    st.title("Image Shear")
    file = st.file_uploader("Choose an image file", type=["jpg", "jpeg", "png"], key="shear_key")
    if not file:
            return
    image = cv2.imdecode(np.frombuffer(file.read(), np.uint8), 1)
    st.write("Please enter a number between decimals (0.05 ~ 1.5)")
    sx = st.number_input("Enter the shear factor along the X axis: ", value=0.0, step=0.05)
    sy = st.number_input("Enter the shear factor along the Y axis: ", value=0.0, step=0.05)
    shear_matrix = np.float32([[1, sx, 0], [sy, 1, 0]])
    sheared_image = cv2.warpAffine(image, shear_matrix, (image.shape[1], image.shape[0]))
    st.image(cv2.cvtColor(sheared_image, cv2.COLOR_BGR2RGB), caption="Image Shear")
    
main()
        