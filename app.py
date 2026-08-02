import os
import json
import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image
from huggingface_hub import hf_hub_download

# ======================================================
# KONFIGURASI HALAMAN
# ======================================================
st.set_page_config(
    page_title="Klasifikasi Sampah CNN",
    page_icon="♻️",
    layout="centered"
)

st.markdown("""
<style>
.stButton>button{
    width:100%;
}
</style>
""", unsafe_allow_html=True)

# ======================================================
# KONFIGURASI MODEL DI HUGGING FACE
# ======================================================
HF_REPO_ID = "mada19/mencoba"
HF_MODEL_FILENAME = "model_klasifikasi_sampah.h5"

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LABEL_MAP_PATH = os.path.join(BASE_DIR, "label_map.json")
IMG_SIZE = (150, 150)

# ======================================================
# LOAD LABEL MAP (tetap dari repo GitHub, file kecil)
# ======================================================
@st.cache_resource
def load_label_map():
    with open(LABEL_MAP_PATH, "r") as f:
        return json.load(f)

label_map = load_label_map()  # contoh: {"0": "Organik", "1": "Anorganik"}

# ======================================================
# DOWNLOAD MODEL DARI HUGGING FACE
# ======================================================
@st.cache_resource
def get_model_path():
    return hf_hub_download(
        repo_id=HF_REPO_ID,
        filename=HF_MODEL_FILENAME
    )

# ======================================================
# LOAD MODEL
# ======================================================
@st.cache_resource
def load_ml_model():
    try:
        model_path = get_model_path()
        model = tf.keras.models.load_model(model_path, compile=False)
        return model
    except Exception as e:
        st.error(f"❌ Gagal memuat model dari Hugging Face:\n\n{e}")
        st.stop()

model = load_ml_model()

# ======================================================
# FUNGSI PREDIKSI
# ======================================================
def predict_image(image: Image.Image):
    img = image.convert("RGB")
    img = img.resize(IMG_SIZE)
    img_array = np.array(img).astype(np.float32) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array, verbose=0)
    confidence = float(prediction[0][0])

    # Output sigmoid: >0.5 -> kelas index 1, <=0.5 -> kelas index 0
    predicted_index = 1 if confidence > 0.5 else 0
    label = label_map[str(predicted_index)]
    score = confidence * 100 if predicted_index == 1 else (1 - confidence) * 100

    return label, score

# ======================================================
# USER INTERFACE
# ======================================================
st.title("♻️ Klasifikasi Sampah Organik & Anorganik")
st.write("Implementasi Deep Learning (CNN) untuk Klasifikasi Citra Sampah Berbasis Web sebagai Upaya Mendukung Smart Environment")

tab1, tab2 = st.tabs(["📂 Upload Gambar", "📷 Kamera"])

# ======================================================
# TAB UPLOAD
# ======================================================
with tab1:
    uploaded_file = st.file_uploader(
        "Pilih gambar sampah...",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Gambar yang diupload", use_column_width=True)

        if st.button("Prediksi", key="btn_upload"):
            with st.spinner("Sedang memproses..."):
                label, score = predict_image(image)

            st.success(f"Hasil Klasifikasi: **{label}**")
            st.progress(min(int(score), 100))
            st.write(f"Confidence: **{score:.2f}%**")

            if label.lower() == "organik":
                st.success("♻️ Sampah ini termasuk kategori **Organik**.")
            else:
                st.warning("♻️ Sampah ini termasuk kategori **Anorganik**.")

# ======================================================
# TAB KAMERA
# ======================================================
with tab2:
    camera_input = st.camera_input("Ambil gambar sampah")

    if camera_input is not None:
        image = Image.open(camera_input).convert("RGB")

        with st.spinner("Sedang memproses..."):
            label, score = predict_image(image)

        st.success(f"Hasil Klasifikasi: **{label}**")
        st.progress(min(int(score), 100))
        st.write(f"Confidence: **{score:.2f}%**")

        if label.lower() == "organik":
            st.success("♻️ Sampah ini termasuk kategori **Organik**.")
        else:
            st.warning("♻️ Sampah ini termasuk kategori **Anorganik**.")

# ======================================================
# FOOTER
# ======================================================
st.markdown("---")
st.caption("Skripsi - Implementasi Deep Learning untuk Klasifikasi Citra Sampah Berbasis Web")
