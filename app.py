
import streamlit as st
import cv2
import numpy as np
from PIL import Image
from io import BytesIO

st.set_page_config(
    page_title="Image Segmentation Toolbox",
    page_icon="🖼️",
    layout="wide"
)

st.title("Image Segmentation Toolbox")
st.write(
    "Region-Based Segmentation, Edge-Based Segmentation "
    "and Object Extraction using Python and OpenCV."
)


def region_segmentation(image, foreground, min_area):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    _, mask = cv2.threshold(
        blurred, 0, 255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    if foreground == "Dark objects":
        mask = cv2.bitwise_not(mask)

    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(
        mask, cv2.MORPH_OPEN, kernel
    )
    mask = cv2.morphologyEx(
        mask, cv2.MORPH_CLOSE, kernel
    )

    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask, connectivity=8
    )

    cleaned = np.zeros_like(mask)

    for i in range(1, count):
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            cleaned[labels == i] = 255

    segmented = cv2.bitwise_and(
        image, image, mask=cleaned
    )

    return cleaned, segmented


def edge_segmentation(image, low, high):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, low, high)
    return edges


def extract_object(image, mask):
    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask, connectivity=8
    )

    if count <= 1:
        return None, None

    largest_label = 1 + np.argmax(
        stats[1:, cv2.CC_STAT_AREA]
    )

    object_mask = np.zeros_like(mask)
    object_mask[labels == largest_label] = 255

    extracted = cv2.bitwise_and(
        image, image, mask=object_mask
    )

    return object_mask, extracted


def image_download(image, filename):
    buffer = BytesIO()
    Image.fromarray(image).save(buffer, format="PNG")
    st.download_button(
        label=f"Download {filename}",
        data=buffer.getvalue(),
        file_name=f"{filename}.png",
        mime="image/png"
    )


uploaded_file = st.file_uploader(
    "Upload an image",
    type=["jpg", "jpeg", "png", "bmp", "tif", "tiff"]
)

with st.sidebar:
    st.header("Segmentation Settings")

    foreground = st.selectbox(
        "Foreground type",
        ["Light objects", "Dark objects"]
    )

    min_area = st.slider(
        "Minimum region area",
        min_value=0,
        max_value=5000,
        value=100,
        step=50
    )

    low = st.slider(
        "Canny lower threshold",
        0, 255, 50
    )

    high = st.slider(
        "Canny upper threshold",
        0, 255, 150
    )

if uploaded_file is not None:
    pil_image = Image.open(uploaded_file).convert("RGB")
    original = np.array(pil_image)

    st.subheader("Original Image")
    st.image(original, use_container_width=True)

    region_tab, edge_tab, extract_tab = st.tabs([
        "Region-Based Segmentation",
        "Edge-Based Segmentation",
        "Object Extraction"
    ])

    with region_tab:
        if st.button("Run Region Segmentation"):
            mask, segmented = region_segmentation(
                original, foreground, min_area
            )

            st.session_state["mask"] = mask
            st.session_state["segmented"] = segmented

        if "mask" in st.session_state:
            col1, col2 = st.columns(2)

            with col1:
                st.image(
                    st.session_state["mask"],
                    caption="Binary Segmentation Mask",
                    use_container_width=True
                )

            with col2:
                st.image(
                    st.session_state["segmented"],
                    caption="Segmented Image",
                    use_container_width=True
                )

            image_download(
                st.session_state["mask"], "segmentation_mask"
            )
            image_download(
                st.session_state["segmented"], "segmented_image"
            )

    with edge_tab:
        if st.button("Run Edge Detection"):
            edges = edge_segmentation(original, low, high)
            st.session_state["edges"] = edges

        if "edges" in st.session_state:
            st.image(
                st.session_state["edges"],
                caption="Canny Edge Detection",
                use_container_width=True
            )
            image_download(
                st.session_state["edges"], "edge_detection"
            )

    with extract_tab:
        st.write(
            "Extract the largest connected object "
            "from the region-based segmentation mask."
        )

        if st.button("Extract Largest Object"):
            mask, _ = region_segmentation(
                original, foreground, min_area
            )
            object_mask, extracted = extract_object(
                original, mask
            )

            if extracted is None:
                st.warning(
                    "No foreground object found. "
                    "Try changing the foreground type or minimum area."
                )
            else:
                st.session_state["object_mask"] = object_mask
                st.session_state["extracted"] = extracted

        if "extracted" in st.session_state:
            col1, col2 = st.columns(2)

            with col1:
                st.image(
                    st.session_state["object_mask"],
                    caption="Extracted Object Mask",
                    use_container_width=True
                )

            with col2:
                st.image(
                    st.session_state["extracted"],
                    caption="Extracted Object",
                    use_container_width=True
                )

            image_download(
                st.session_state["extracted"], "extracted_object"
            )

else:
    st.info("Upload an image to start segmentation.")
