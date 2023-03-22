from io import BytesIO
from time import time


from PIL import Image
import streamlit as st
import numpy as np
import cv2


from panorama.stitcher import Stitcher
from fuzzybach import fuzzy_contrast_enhance
# from fuzzy.image_enhancement import ImageEnhancement

def plot_image_grid(images):
    '''Plot a grid of images'''
    if len(images) == 2:
        number_of_columns = 2
    else:
        number_of_columns = 4
    number_of_rows = len(images) // number_of_columns + len(images) % number_of_columns
    total_columns = []

    # print('Number of rows:', number_of_rows)
    # print('Number of columns:', number_of_columns)

    for i in range(0, number_of_rows):
        columns = st.columns(number_of_columns)
        total_columns.extend(columns)

    for i, (col, image) in enumerate(zip(total_columns, images), start=1):
        col.image(image, f'Image {i}')


@st.cache_data
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

    panorama = Stitcher()
    # member_name()

    st.title(':sun_with_face: Group 2 - IMP301')
    st.title('Topic: Panorama Image')
    
    # Upload images
    file_upload = st.sidebar.file_uploader(':point_down: **Upload images**', accept_multiple_files=True)

    # User choices
    panorama_status = st.sidebar.checkbox('Create panorama')

    fuzzy_status = st.sidebar.radio('Fuzzy enhancement',
                                    ('No enhancement', 'Enhance input', 'Enhance output', 'Enhance both'))
    # fuzzy = st.sidebar.checkbox('Fuzzy enhancement input images')
    # fuzzy = st.sidebar.checkbox('Fuzzy enhancement panorama images')

    number_uploaded_images = len(file_upload)
    imgs_uploaded = []
    fuzzy_imgs = []

    if file_upload is None or number_uploaded_images == 0:
        background_img = read_image('images/Big-rock-nature-computer-backgrounds.jpg')
        st.image(background_img)
    elif number_uploaded_images < 2: 
        st.error('Please upload at least 2 images', icon="🚨")
    else:
        for i, f in enumerate(file_upload):
            img = read_image(f)
            fuzzy_imgs.append(fuzzy_contrast_enhance(img))
            imgs_uploaded.append(img)

        st.info('Input images', icon="ℹ️")
        plot_image_grid(imgs_uploaded)

        if fuzzy_status   == 'Enhance input':
            st.info('Fuzzy input images', icon="ℹ️")
            plot_image_grid(fuzzy_imgs)

    # Panorama
    if len(imgs_uploaded) >= 2 and panorama_status:
        st.success('Result', icon="✅")

        if fuzzy_status == 'No enhancement':
            start_pano = time()
            panorama_img = panorama.stitch(imgs_uploaded)
            end_pano = time()
            print('Panorama time:', end_pano - start_pano)
            cv2.imwrite('images/results/no_enh_panorama.jpg', panorama_img[:, :, ::-1])

            st.image(panorama_img, caption='Panorama')

        elif fuzzy_status   == 'Enhance input':
            start_pano = time()
            panorama_img = panorama.stitch(fuzzy_imgs)
            end_pano = time()
            print('Panorama time:', end_pano - start_pano)
            cv2.imwrite('images/results/enh_inp_panorama.jpg', panorama_img[:, :, ::-1])

            st.image(panorama_img, caption='Panorama')

        elif fuzzy_status == 'Enhance output':
            panorama_img = panorama.stitch(imgs_uploaded)
            start_fuzzy = time()
            fuzz_img = fuzzy_contrast_enhance(panorama_img)
            end_fuzzy = time()
            print('Fuzzy time:', end_fuzzy - start_fuzzy)
            cv2.imwrite('images/results/enh_out_panorama.jpg', panorama_img[:, :, ::-1])

            st.image(panorama_img, caption='Panorama')
            st.image(fuzz_img, caption='Fuzzy enhance')

        elif fuzzy_status == 'Enhance both':
            panorama_img = panorama.stitch(fuzzy_imgs)
            start_fuzzy = time()
            fuzz_img = fuzzy_contrast_enhance(panorama_img)
            end_fuzzy = time()
            print('Fuzzy time:', end_fuzzy - start_fuzzy)
            cv2.imwrite('images/results/both.jpg', fuzz_img[:, :, ::-1])
            
            st.image(fuzz_img, caption='Fuzzy enhance')

