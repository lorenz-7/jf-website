import streamlit as st
from streamlit_option_menu import option_menu
from streamlit_embedcode import github_gist
from data import *
from model_helper.helper_functions import *
from model_helper.arch import *
from model_helper.metrics import *
from PIL import Image
import keras.backend as K
import cv2

with st.sidebar:
    selection_screen = option_menu("Seize Disease", ["Home", "About", "Liver", "Melanoma", "Tract"],
                                   icons=["house", "info", "bandaid-fill", "bandaid-fill", "bandaid-fill"],
                                   menu_icon="file-earmark-medical-fill")

if selection_screen == "Home":
    st.title("Home")


elif selection_screen == "About":
    st.title("About")

elif selection_screen == "Liver":
    button = st.sidebar.button("Bild Segmentieren")
    st.sidebar.subheader("Einstellungen")
    model = st.sidebar.selectbox("Modell", ["AttResUnet"])
    upload_settings = st.selectbox("Upload Settings", ["Test Bilder", "Eigenes Bild"])
    col1, col2, col3 = st.columns(3)
    if upload_settings == "Test Bilder":
        with col1:
            test_img = st.radio("Test Bilder", ["Bild 1", "Bild 2", "Bild 3"])
            if test_img == "Bild 1":
                img = Image.open("./data/test_liver1.jpg")
                st.image(img, width=400, caption="Test Bild 1")
            elif test_img == "Bild 2":
                img = Image.open("./data/test_liver2.jpg")
                st.image(img, width=400, caption="Test Bild 2")
            elif test_img == "Bild 3":
                img = Image.open("./data/test_liver3.jpg")
                st.image(img, width=400, caption="Test Bild 3")
        with col2:
            color_selection = st.radio("Farbauswahl", ["Hot", "Ocean", "Pink"])
            if color_selection == "Hot":
                color = cv2.COLORMAP_HOT
            elif color_selection == "Ocean":
                color = cv2.COLORMAP_OCEAN
            elif color_selection == "Pink":
                color = cv2.COLORMAP_PINK
        with col3:
            channels = st.radio("Kanäle", ["BGR", "RGB"])
            if channels == "BGR":
                channels = None
            elif channels == "RGB":
                channels = cv2.COLOR_BGR2RGB
        if button and img is not None:
            model_name = "models/AttResUnet-Liver.h5"
            model = load(model_name,
                         custom_objects={"dice_score": dice_score, "f1_m": f1_m, "precision_m": precision_m,
                                         "recall_m": recall_m, "K": K})
            mask = predict(img, model, type="test")
            st.image(mask, width=400, caption="Segmentiertes Bild", clamp=True)
            color_mask = colorize_mask(mask, color, channels)
            st.image(color_mask, width=400, caption="Farbiges, Segmentiertes Bild", clamp=True)



    elif upload_settings == "Eigenes Bild":
        uploaded_file = st.file_uploader("Bild hochladen", type=["jpg", "png"])
        st.warning(
            "Bilder, die aus dem UWMGl Datenset stammen, kann das Modell am besten klassifizieren. Natürlich können "
            "Sie auch andere Bilder hochladen.")
        col4, col5 = st.columns(2)
        with col4:
            color_selection = st.radio("Farbauswahl", ["Hot", "Ocean", "Pink"])
        if color_selection == "Hot":
            color = cv2.COLORMAP_HOT
        elif color_selection == "Ocean":
            color = cv2.COLORMAP_JET
        elif color_selection == "Pink":
            color = cv2.COLORMAP_COOL
        with col5:
            channels = st.radio("Kanäle", ["BGR", "RGB"])
            if channels == "BGR":
                channels = None
            elif channels == "RGB":
                channels = cv2.COLOR_BGR2RGB
        if uploaded_file is not None:
            img = Image.open(uploaded_file)
            st.image(img, width=400, caption="Eigenes Bild")
            img = img.resize((128, 128))
            img = make_gray(img, "BGR")

            if button and img is not None:
                model_name = "models/AttResUnet-Liver.h5"
                model = load(model_name,
                             custom_objects={"dice_score": dice_score, "f1_m": f1_m, "precision_m": precision_m,
                                             "recall_m": recall_m, "K": K})
                mask = predict(img, model, type=None)
                st.image(mask, width=400, caption="Segmentiertes Bild", clamp=True)
                color_mask = colorize_mask(mask, color, channels)
                st.image(color_mask, width=400, caption="Farbiges, Segmentiertes Bild", clamp=True)



elif selection_screen == "Melanoma":
    st.sidebar.subheader("Einstellungen")
    model = st.sidebar.selectbox("Modell", ["Swin-Mid", "Swin-Large"])
    upload_settings = st.selectbox("Upload Settings", ["Test Bilder", "Eigenes Bild", "Kamera"])
    col1, col2, col3 = st.columns(3)
    button = st.sidebar.button("Bild Segmentieren")
    if upload_settings == "Test Bilder":
        with col1:
            test_img = st.radio("Test Bilder", ["Bild 1", "Bild 2", "Bild 3"])
            if test_img == "Bild 1":
                img = Image.open("./data/test_image1.jpg")
                st.image(img, width=400, caption="Test Bild 1")
            elif test_img == "Bild 2":
                img = Image.open("./data/test_image2.jpg")
                st.image(img, width=400, caption="Test Bild 2")
            elif test_img == "Bild 3":
                img = Image.open("./data/test_image3.jpg")
                st.image(img, width=400, caption="Test Bild 3")
        with col2:
            color_selection = st.radio("Farbauswahl", ["Hot", "Ocean", "Pink"])
            if color_selection == "Hot":
                color = cv2.COLORMAP_HOT
            elif color_selection == "Ocean":
                color = cv2.COLORMAP_OCEAN
            elif color_selection == "Pink":
                color = cv2.COLORMAP_PINK
        with col3:
            channels = st.radio("Kanäle", ["BGR", "RGB"])
            if channels == "BGR":
                channels = None
            elif channels == "RGB":
                channels = cv2.COLOR_BGR2RGB

    elif upload_settings == "Eigenes Bild":
        uploaded_file = st.file_uploader("Bild hochladen", type=["jpg", "png"])
        col4, col5 = st.columns(2)
        with col4:
            color_selection = st.radio("Farbauswahl", ["Hot", "Ocean", "Pink"])
            if color_selection == "Hot":
                color = cv2.COLORMAP_HOT
            elif color_selection == "Ocean":
                color = cv2.COLORMAP_OCEAN
            elif color_selection == "Pink":
                color = cv2.COLORMAP_PINK
        with col5:
            channels = st.radio("Kanäle", ["BGR", "RGB"])
            if channels == "BGR":
                channels = None
            elif channels == "RGB":
                channels = cv2.COLOR_BGR2RGB
        if uploaded_file is not None:
            img = Image.open(uploaded_file)
            st.image(img, width=400, caption="Eigenes Bild")
            img = img.resize((128, 128))
            img = make_gray(img, "BGR")



    elif upload_settings == "Kamera":
        uploaded_file = st.camera_input("Kamera", key="camera")
        col4, col5 = st.columns(2)
        with col4:
                color_selection = st.radio("Farbauswahl", ["Hot", "Ocean", "Pink"])
                if color_selection == "Hot":
                    color = cv2.COLORMAP_HOT
                elif color_selection == "Ocean":
                    color = cv2.COLORMAP_OCEAN
                elif color_selection == "Pink":
                    color = cv2.COLORMAP_PINK
        with col5:
                channels = st.radio("Kanäle", ["BGR", "RGB"])
                if channels == "BGR":
                    channels = None
                elif channels == "RGB":
                    channels = cv2.COLOR_BGR2RGB
        if uploaded_file is not None:
            img = Image.open(uploaded_file)
            st.image(img, width=400, caption="Eigenes Bild")
            img = img.resize((128, 128))
            img = make_gray(img, "BGR")
    if button and img is not None:
        if model == "Swin-Mid":
            model_name = "models/Swin-Mid.h5"
        elif model == "Swin-Large":
            model_name = "models/Swin-Large.h5"
        model = load(model_name,
                     custom_objects={'patch_extract': patch_extract, 'patch_merging': patch_merging,
                                     'dice_score': dice_score, 'recall_m': recall_m, 'precision_m': precision_m,
                                     'f1_m': f1_m, 'patch_embedding': patch_embedding, 'patch_expanding':
                                         patch_expanding, 'K': K, 'SwinTransformerBlock': SwinTransformerBlock})
        mask = predict(img, model, type=None)
        st.image(mask, width=400, caption="Segmentiertes Bild", clamp=True)
        color_mask = colorize_mask(mask, color, channels)
        st.image(color_mask, width=400, caption="Farbiges, Segmentiertes Bild", clamp=True)

elif selection_screen == "Tract":
    st.sidebar.subheader("Einstellungen")
    model = st.sidebar.selectbox("Modell", ["AttResUnet"])
    upload_settings = st.selectbox("Upload Settings", ["Test Bilder", "Eigenes Bild"])
    col1, col2, col3 = st.columns(3)
    button = st.sidebar.button("Bild Segmentieren")
    if upload_settings == "Test Bilder":
        with col1:
            test_img = st.radio("Test Bilder", ["Bild 1", "Bild 2", "Bild 3"])
            if test_img == "Bild 1":
                img = Image.open("./data/test_tract1.png")
                st.image(img, width=400, caption="Test Bild 1")
            elif test_img == "Bild 2":
                img = Image.open("./data/test_tract2.png")
                st.image(img, width=400, caption="Test Bild 2")
            elif test_img == "Bild 3":
                img = Image.open("./data/test_tract3.png")
                st.image(img, width=400, caption="Test Bild 3")
        with col2:
            color_selection = st.radio("Farbauswahl", ["Hot", "Ocean", "Pink"])
            if color_selection == "Hot":
                color = cv2.COLORMAP_HOT
            elif color_selection == "Ocean":
                color = cv2.COLORMAP_OCEAN
            elif color_selection == "Pink":
                color = cv2.COLORMAP_PINK
        with col3:
            channels = st.radio("Kanäle", ["BGR", "RGB"])
            if channels == "BGR":
                channels = None
            elif channels == "RGB":
                channels = cv2.COLOR_BGR2RGB
        if button and img is not None:
            if model == "AttResUnet":
                model_name = "models/AttResUnet-Tract.h5"
            model = load(model_name, custom_objects={"dice_score": dice_score, "f1_m": f1_m, "precision_m": precision_m,
                                         "recall_m": recall_m, "K": K} )
            mask = predict(img, model, type="test")
            st.image(mask, width=400, caption="Segmentiertes Bild", clamp=True)
            color_mask = colorize_mask(mask, color, channels)
            st.image(color_mask, width=400, caption="Farbiges, Segmentiertes Bild", clamp=True)

    elif upload_settings == "Eigenes Bild":
        uploaded_file = st.file_uploader("Bild hochladen", type=["jpg", "png"])
        col4, col5 = st.columns(2)
        with col4:
                color_selection = st.radio("Farbauswahl", ["Hot", "Ocean", "Pink"])
                if color_selection == "Hot":
                    color = cv2.COLORMAP_HOT
                elif color_selection == "Ocean":
                    color = cv2.COLORMAP_OCEAN
                elif color_selection == "Pink":
                    color = cv2.COLORMAP_PINK
        with col5:
                channels = st.radio("Kanäle", ["BGR", "RGB"])
                if channels == "BGR":
                    channels = None
                elif channels == "RGB":
                    channels = cv2.COLOR_BGR2RGB
        if uploaded_file is not None:
            img = Image.open(uploaded_file)
            st.image(img, width=400, caption="Eigenes Bild")
            img = img.resize((128, 128))
            img = make_gray(img, "RGB")
        if button and img is not None:
            if model == "AttResUnet":
                model_name = "models/AttResUnet-Tract.h5"
            model = load(model_name, custom_objects={"dice_score": dice_score, "f1_m": f1_m, "precision_m": precision_m,
                                         "recall_m": recall_m, "K": K})
            mask = predict(img, model, type=None)
            st.image(mask, width=400, caption="Segmentiertes Bild", clamp=True)
            color_mask = colorize_mask(mask, color, channels)
            st.image(color_mask, width=400, caption="Farbiges, Segmentiertes Bild", clamp=True)


