"""Classify page: take or upload a photo, save it, and run it through the chosen CNN."""

import hashlib
import mimetypes
import os
import random
import time

import streamlit as st
from PIL import Image

from app_pages import common
from vision import charts, inference

mobile = common.is_mobile()
spec = common.current_spec()

common.hero(
    "CNN Vision",
    "Take a photo or upload one, then let a convolutional neural network tell you what it sees.",
    icon="🔮",
)

# ---------- 1. Model ----------
st.markdown('<div class="step">1 · Choose a model</div>', unsafe_allow_html=True)
spec = common.model_menu("Model")
lm = common.get_model(spec.id)

if mobile:
    with st.expander("⚙️ Settings"):
        threshold = common.threshold_slider()
        face_crop = common.face_crop_toggle(spec)
else:
    with st.sidebar:
        threshold = common.threshold_slider()
        face_crop = common.face_crop_toggle(spec)
    common.sidebar_footer()

# ---------- 2. Image ----------
st.markdown('<div class="step">2 · Add a photo</div>', unsafe_allow_html=True)

SOURCES = {
    "camera": "📷 Camera",
    "gallery": "🖼️ Gallery",
    "computer": "💻 Computer",
    "sample": "🎲 Sample",
}
if mobile:
    # On a phone, Gallery also opens the phone's files
    del SOURCES["computer"]
    if st.session_state.get("source") == "computer":
        st.session_state.source = "gallery"
st.segmented_control("Where is the photo?", list(SOURCES), format_func=SOURCES.get, required=True,
                     width="stretch", label_visibility="collapsed", **common.bind("source", "camera"))
source = st.session_state.source

image_bytes, file_name, true_class = None, None, None
if source == "camera":
    shot = st.camera_input("Take a picture (phone or computer camera)",
                           help="Your browser will ask for permission to use the camera.")
    if shot:
        image_bytes, file_name = shot.getvalue(), f"photo_{time.strftime('%Y-%m-%d_%H-%M-%S')}.jpg"
elif source in ("gallery", "computer"):
    label = ("Choose a photo from your gallery" if source == "gallery"
             else "Upload an image file from your computer")
    tip = ("On a phone this opens your photo gallery (you can also take a new photo with the phone's camera app)."
           if source == "gallery" else "Drag and drop a file here, or click Browse files.")
    upload = st.file_uploader(label, type=["jpg", "jpeg", "png", "webp", "bmp"], key=f"upload_{source}", help=tip)
    st.caption(tip)
    if upload:
        image_bytes, file_name = upload.getvalue(), upload.name
else:
    # A random image from the test set the model was evaluated on - or, without the dataset
    # (e.g. in a fresh clone of the repo), from the few test images kept in the samples folder
    files = st.session_state.setdefault("_sample_files", {})
    if spec.id not in files:
        files[spec.id] = (inference.list_images(spec.test_dir, lm.classes)
                          or inference.list_images(spec.samples_dir, lm.classes))
    items = files[spec.id]
    if not items:
        st.warning("There are no sample images for this model.")
    else:
        sample_key = f"sample_{spec.id}"
        if st.button("🎲 Pick another sample", width="stretch") or sample_key not in st.session_state:
            st.session_state[sample_key] = random.choice(items)
        path, class_index = st.session_state[sample_key]
        with open(path, "rb") as f:
            image_bytes = f.read()
        file_name, true_class = os.path.basename(path), lm.classes[class_index]
        st.caption(f"A random photo from the test set · real answer: **{spec.label(true_class)}**")

if image_bytes is None:
    st.markdown(
        '<div class="empty-state"><div class="big">🖼️</div>'
        "<b>No photo yet</b><br>Take a picture, choose one from your gallery or computer, or try a sample.</div>",
        unsafe_allow_html=True,
    )
    st.stop()

try:
    image = inference.open_image(image_bytes)
except Exception:
    st.error("That file could not be opened as an image. Please try a JPG or PNG.")
    st.stop()

image_id = hashlib.sha1(image_bytes).hexdigest()

# ---------- 3. Buttons + result ----------
if mobile:
    preview_area, result_area = st.container(), st.container()
else:
    preview_area, result_area = st.columns([5, 6], gap="large")

with preview_area:
    st.image(image, width="stretch")
    classify_col, save_col = st.columns(2)
    with classify_col:
        clicked = st.button(f"{spec.icon} Classify", type="primary", width="stretch")
    with save_col:
        st.download_button(
            "💾 Save image", data=image_bytes, file_name=file_name,
            mime=mimetypes.guess_type(file_name)[0] or "image/jpeg", width="stretch",
            on_click="ignore",
        )

if clicked:
    st.session_state.classified = (image_id, spec.id)

with result_area:
    if st.session_state.get("classified") != (image_id, spec.id):
        st.markdown('<div class="step">3 · Classify</div>', unsafe_allow_html=True)
        st.info(f"Press **{spec.icon} Classify** to run the photo through the {spec.title} model.", icon="👈" if not mobile else "☝️")
        st.stop()

    # Runs again when a setting changes, so the result always matches the settings (it takes a few milliseconds)
    result = inference.predict(lm, image, threshold=threshold, face_crop=face_crop)

    if result.is_unknown:
        best = spec.label(result.class_name)
        card = (
            f'<div class="result-card unknown"><div class="result-kicker">Result</div>'
            f'<div class="result-label">❓ Unknown</div>'
            f'<div class="result-meter"><div style="width:{result.confidence:.0%}"></div></div>'
            f'<div class="result-meta">The model is only <b>{result.confidence:.0%}</b> sure, which is below the '
            f'{threshold:.0%} needed. Best guess: <b>{best}</b>.</div></div>'
        )
    else:
        card = (
            f'<div class="result-card"><div class="result-kicker">The model sees</div>'
            f'<div class="result-label">{result.label}</div>'
            f'<div class="result-meter"><div style="width:{result.confidence:.0%}"></div></div>'
            f'<div class="result-meta">Confidence: <b>{result.confidence:.1%}</b></div></div>'
        )
    st.markdown(card, unsafe_allow_html=True)

    if true_class is not None:
        if result.class_name == true_class and not result.is_unknown:
            st.success(f"Correct! The real answer is {spec.label(true_class)}.", icon="✅")
        elif result.class_name == true_class:
            st.warning(f"The best guess is right ({spec.label(true_class)}), but the model wasn't sure enough.", icon="🤔")
        else:
            st.error(f"Not quite - the real answer is {spec.label(true_class)}.", icon="❌")
    for note in result.notes:
        st.warning(note, icon="🔎")

    st.markdown("**Probability of each option**")
    st.plotly_chart(charts.probability_chart(result, spec, compact=mobile), width="stretch",
                    config={"displayModeBar": False})

    with st.expander("🔬 What the model actually saw"):
        height, width = lm.image_size
        big = result.model_input.resize((width * 4, height * 4), Image.NEAREST)
        cols = st.columns([1, 2])
        cols[0].image(big, width="stretch")
        cols[1].markdown(
            f"The photo was prepared exactly like the training images:\n"
            f"1. {'Cropped to the face, then r' if result.face_found else 'R'}esized to **{width}×{height}** pixels"
            f"{' in grayscale' if lm.color_mode == 'grayscale' else ''}\n"
            f"2. Pixel values scaled from 0-255 to **0-1**\n"
            f"3. Shaped as a batch of one: **(1, {height}, {width}, {lm.channels})**"
        )
