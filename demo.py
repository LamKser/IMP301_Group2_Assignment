from io import BytesIO

from PIL import Image
import streamlit as st
import numpy as np
import cv2


from panorama.stitcher import Stitcher
# from fuzzy import fuzzy_contrast_enhance
from fuzzy.image_enhancement import ImageEnhancement

def plot_image_grid(images):
    '''Plot a grid of images'''
    if len(images) == 2:
        number_of_columns = 2
    else:
        number_of_columns = 4
    number_of_rows = int(len(images) / number_of_columns) + int(len(images) % number_of_columns)
    total_columns = []

    # print('Number of rows:', number_of_rows)
    # print('Number of columns:', number_of_columns)

    for i in range(0, number_of_rows):
        columns = st.columns(number_of_columns)
        total_columns.extend(columns)

    for i, (col, image) in enumerate(zip(total_columns, images), start=1):
        col.image(image, f'Image {i}')


def read_image(file):
    image = Image.open(file)
    image_array = np.array(image)
    return image_array


def member_name():
    with st.sidebar:
        st.write('Đinh Hoàng Lâm')
        st.write('Trần Duy Ngọc Bảo')
        st.write('Nguyễn Gia Bách')
        st.write('Nguyễn Cao Trí')
        st.write('Đặng Chí Thanh')
        st.write('Dương Viễn Thạch')


if __name__ == '__main__':

    panorama = Stitcher(**{"detector": "sift", "confidence_threshold": 0.2})
    # member_name()

    st.title('Group 2 - IMP301 - Image processing')
    st.title('Topic: Panorama Image')
    
    # Upload images
    file_upload = st.sidebar.file_uploader('Upload images', accept_multiple_files=True)
    fuzzy = st.sidebar.checkbox('Fuzzy enhancement')

    number_uploaded_images = len(file_upload)
    imgs_uploaded = []

    if file_upload is None or number_uploaded_images == 0:
        background_img = read_image('images/Big-rock-nature-computer-backgrounds.jpg')
        st.image(background_img)
    elif number_uploaded_images < 2: 
        st.error('Please upload at least 2 images', icon="🚨")
    else:
        for i, f in enumerate(file_upload):
            img = read_image(f)
            imgs_uploaded.append(img)
        st.info('Input images', icon="ℹ️")
        plot_image_grid(imgs_uploaded)

    # Panorama
    if len(imgs_uploaded) >= 2:
        st.success('Panorama result', icon="✅")
        panorama_img = panorama.stitch(imgs_uploaded)
        st.image(panorama_img)
        if fuzzy:
            # st.image(fuzzy_contrast_enhance(panorama_img))
            img_enh = ImageEnhancement(panorama_img, 2, 2, 1, True)
            result = img_enh.enhance_colored_img()
            st.image(result)

    # Fuzzy
    
    # if fuzzy:
    #     st.write('Off fuzzy')
    # else:
    #     st.write('')