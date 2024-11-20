import streamlit as st
import cv2
import numpy as np
from io import BytesIO
from PIL import Image

# Functions for segmentation techniques
@st.cache_data
def threshold_segmentation(image):
    _, global_thresh = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)
    adaptive_thresh = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    return global_thresh, adaptive_thresh

@st.cache_data
def kmeans_segmentation(image):
    data = image.reshape((-1, 3)).astype(np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    _, labels, centers = cv2.kmeans(data, 3, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    segmented_image = centers[labels.flatten()].reshape(image.shape).astype(np.uint8)
    return segmented_image

@st.cache_data
def watershed_segmentation(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    kernel = np.ones((3, 3), np.uint8)
    sure_bg = cv2.dilate(thresh, kernel, iterations=3)
    dist_transform = cv2.distanceTransform(thresh, cv2.DIST_L2, 5)
    _, sure_fg = cv2.threshold(dist_transform, 0.7 * dist_transform.max(), 255, 0)
    sure_fg = np.uint8(sure_fg)
    unknown = cv2.subtract(sure_bg, sure_fg)
    _, markers = cv2.connectedComponents(sure_fg)
    markers += 1
    markers[unknown == 255] = 0
    markers = cv2.watershed(image, markers)
    image[markers == -1] = [255, 0, 0]
    return image

@st.cache_data
def grabcut_segmentation(image):
    mask = np.zeros(image.shape[:2], np.uint8)
    rect = (50, 50, image.shape[1] - 100, image.shape[0] - 100)
    bgd_model = np.zeros((1, 65), np.float64)
    fgd_model = np.zeros((1, 65), np.float64)
    cv2.grabCut(image, mask, rect, bgd_model, fgd_model, 5, cv2.GC_INIT_WITH_RECT)
    mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
    segmented_image = image * mask2[:, :, np.newaxis]
    return segmented_image

# Streamlit App
st.title("Image Segmentation App")

# Sidebar for navigation
st.sidebar.title("Navigation")
options = ["Upload Image", "Threshold Segmentation", "K-Means Segmentation", "Watershed Segmentation", "GrabCut Segmentation"]
selected_option = st.sidebar.radio("Choose a segmentation technique:", options)

# Upload Image
if selected_option == "Upload Image":
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])
    if uploaded_file:
        image = np.array(Image.open(uploaded_file))
        st.image(image, caption="Uploaded Image", use_container_width=True)
        st.session_state["image"] = image

# Perform Segmentation
if "image" in st.session_state:
    image = st.session_state["image"]

    if selected_option == "Threshold Segmentation":
        st.header("Threshold Segmentation")
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        global_thresh, adaptive_thresh = threshold_segmentation(gray_image)
        st.image(global_thresh, caption="Global Threshold", channels="GRAY", use_container_width=True)
        st.download_button("Download Global Threshold", data=cv2.imencode('.jpg', global_thresh)[1].tobytes(), file_name="global_threshold.jpg")
        st.image(adaptive_thresh, caption="Adaptive Threshold", channels="GRAY", use_container_width=True)
        st.download_button("Download Adaptive Threshold", data=cv2.imencode('.jpg', adaptive_thresh)[1].tobytes(), file_name="adaptive_threshold.jpg")

    elif selected_option == "K-Means Segmentation":
        st.header("K-Means Segmentation")
        kmeans_result = kmeans_segmentation(image)
        st.image(kmeans_result, caption="K-Means Segmentation", channels="BGR", use_container_width=True)
        st.download_button("Download K-Means Segmentation", data=cv2.imencode('.jpg', kmeans_result)[1].tobytes(), file_name="kmeans_segmentation.jpg")

    elif selected_option == "Watershed Segmentation":
        st.header("Watershed Segmentation")
        watershed_result = watershed_segmentation(image.copy())
        st.image(watershed_result, caption="Watershed Segmentation", channels="BGR", use_container_width=True)
        st.download_button("Download Watershed Segmentation", data=cv2.imencode('.jpg', watershed_result)[1].tobytes(), file_name="watershed_segmentation.jpg")

    elif selected_option == "GrabCut Segmentation":
        st.header("GrabCut Segmentation")
        grabcut_result = grabcut_segmentation(image.copy())
        st.image(grabcut_result, caption="GrabCut Segmentation", channels="BGR", use_container_width=True)
        st.download_button("Download GrabCut Segmentation", data=cv2.imencode('.jpg', grabcut_result)[1].tobytes(), file_name="grabcut_segmentation.jpg")
