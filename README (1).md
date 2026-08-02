# ♻️ Klasifikasi Sampah Organik & Anorganik (CNN + Streamlit)

Skripsi: **Implementasi Deep Learning untuk Klasifikasi Citra Sampah Organik dan Anorganik
Menggunakan Metode Convolutional Neural Network Berbasis Web sebagai Upaya Mendukung Smart Environment**

## Struktur File
```
├── app.py                          # Aplikasi Streamlit
├── model_klasifikasi_sampah.h5     # Model CNN hasil training
├── label_map.json                  # Mapping label (0: Organik, 1: Anorganik)
├── requirements.txt                # Dependency Python
├── runtime.txt                     # Versi Python untuk Streamlit Cloud
└── README.md
```

## Cara Menjalankan Lokal
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Cara Deploy ke Streamlit Community Cloud
1. Push seluruh isi folder ini ke repository GitHub (public/private).
2. Buka https://share.streamlit.io dan login dengan akun GitHub.
3. Klik **New app**, pilih repository ini.
4. Pastikan **Main file path** diisi `app.py`.
5. Klik **Deploy**.

Model (`model_klasifikasi_sampah.h5`, ±14MB) sudah cukup kecil untuk langsung di-push ke GitHub
tanpa perlu hosting eksternal seperti Hugging Face.

## Author
Skripsi Klasifikasi Citra Sampah Berbasis Web — 2026
