from pathlib import Path

import cv2
import numpy as np
import streamlit as st

st.set_page_config(page_title="Deteksi Tepi Citra", page_icon="🕷️", layout="wide")

# Batas sisi terpanjang citra agar proses tetap cepat di server
MAKS_SISI = 1600

# ---------------------------------------------------------------
# Konfigurasi operator (warna & deskripsi)
# ---------------------------------------------------------------
OPERATOR = {
    "Sobel":   {"warna": "#3b82f6", "info": "Kernel 3x3, bobot 2 di tengah. Tepi halus & tahan noise."},
    "Prewitt": {"warna": "#10b981", "info": "Kernel 3x3, bobot sama rata. Mirip Sobel, lebih sederhana."},
    "Roberts": {"warna": "#f59e0b", "info": "Kernel 2x2, selisih diagonal. Cepat, tetapi sensitif noise."},
    "Canny":   {"warna": "#ec4899", "info": "Blur + NMS + hysteresis. Tepi tipis dan bersih."},
}

# CSS: beri warna tiap tombol berdasarkan key (class .st-key-<key>)
css = ""
for nama, o in OPERATOR.items():
    css += f"""
    .st-key-btn_{nama} button {{
        background: {o['warna']}; color: white; border: none;
        height: 3.4rem; font-size: 1.05rem; font-weight: 700;
        border-radius: 12px; transition: all .15s ease;
    }}
    .st-key-btn_{nama} button:hover {{
        transform: translateY(-3px);
        box-shadow: 0 8px 20px {o['warna']}66; color: white;
    }}
    """
st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)


# ---------------------------------------------------------------
# Fungsi bantu & operator deteksi tepi
# ---------------------------------------------------------------
def magnitudo_ke_uint8(gx, gy):
    return np.clip(cv2.magnitude(gx, gy), 0, 255).astype(np.uint8)


def proses_sobel(gray):
    sx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    sy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    hasil = magnitudo_ke_uint8(sx, sy)
    return [(cv2.convertScaleAbs(sx), "Gradient X"),
            (cv2.convertScaleAbs(sy), "Gradient Y"),
            (hasil, "Sobel")], hasil


def proses_prewitt(gray):
    kx = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]], dtype=np.float32)
    ky = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]], dtype=np.float32)
    px = cv2.filter2D(gray, cv2.CV_32F, kx)
    py = cv2.filter2D(gray, cv2.CV_32F, ky)
    hasil = magnitudo_ke_uint8(px, py)
    return [(cv2.convertScaleAbs(px), "Gradient X"),
            (cv2.convertScaleAbs(py), "Gradient Y"),
            (hasil, "Prewitt")], hasil


def proses_roberts(gray):
    kx = np.array([[1, 0], [0, -1]], dtype=np.float32)
    ky = np.array([[0, 1], [-1, 0]], dtype=np.float32)
    rx = cv2.filter2D(gray, cv2.CV_32F, kx)
    ry = cv2.filter2D(gray, cv2.CV_32F, ky)
    hasil = magnitudo_ke_uint8(rx, ry)
    return [(cv2.convertScaleAbs(rx), "Gradient X (diagonal)"),
            (cv2.convertScaleAbs(ry), "Gradient Y (diagonal)"),
            (hasil, "Roberts")], hasil


def proses_canny(gray, t1_rendah, t2_rendah, t1_tinggi, t2_tinggi):
    blur = cv2.GaussianBlur(gray, (5, 5), sigmaX=1.4)
    c_rendah = cv2.Canny(blur, t1_rendah, t2_rendah)
    c_tinggi = cv2.Canny(blur, t1_tinggi, t2_tinggi)
    return [(blur, "Grayscale + Gaussian Blur"),
            (c_rendah, f"Canny ({t1_rendah}, {t2_rendah})"),
            (c_tinggi, f"Canny ({t1_tinggi}, {t2_tinggi})")], c_rendah


def decode_citra(file_obj):
    """Decode berkas unggahan/kamera menjadi array BGR (None bila gagal)."""
    data = np.frombuffer(file_obj.getvalue(), dtype=np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def path_contoh():
    path = Path(__file__).resolve().parent / "spider.jpeg"
    return path if path.exists() else None


# ---------------------------------------------------------------
# Sidebar (parameter)
# ---------------------------------------------------------------
with st.sidebar:
    st.header("Pengaturan")
    st.subheader("Parameter Canny")
    t1a, t2a = st.slider("Pasangan 1 (lebih banyak tepi)", 0, 500, (50, 100))
    t1b, t2b = st.slider("Pasangan 2 (hanya tepi kuat)", 0, 500, (100, 200))

# ---------------------------------------------------------------
# Halaman utama
# ---------------------------------------------------------------
st.title("Deteksi Tepi Citra")
st.caption("Unggah gambar, lalu pilih operator untuk memprosesnya.")

# ----- Langkah 1: sumber citra -----
st.subheader("1. Pilih citra")

sumber_opsi = ["Unggah gambar", "Ambil dari kamera"]
if path_contoh() is not None:
    sumber_opsi.append("Gunakan contoh (spider.jpeg)")

sumber = st.radio("Sumber citra", sumber_opsi, horizontal=True, label_visibility="collapsed")

bgr = None
nama_citra = ""

if sumber == "Unggah gambar":
    unggahan = st.file_uploader(
        "Seret & lepas gambar di sini, atau klik Browse files",
        type=["jpg", "jpeg", "png", "bmp", "webp"],
        help="Format yang didukung: JPG, JPEG, PNG, BMP, WEBP",
    )
    if unggahan is not None:
        bgr = decode_citra(unggahan)
        nama_citra = unggahan.name
elif sumber == "Ambil dari kamera":
    foto = st.camera_input("Ambil foto")
    if foto is not None:
        bgr = decode_citra(foto)
        nama_citra = "foto_kamera.png"
else:
    bgr = cv2.imread(str(path_contoh()))
    nama_citra = "spider.jpeg"

if bgr is None:
    if sumber == "Unggah gambar" and unggahan is not None:
        st.error("Gambar gagal dibaca. Pastikan file valid dan formatnya didukung.")
    else:
        st.info("Belum ada citra. Unggah gambar, ambil foto, atau pilih contoh untuk memulai.")
    st.stop()

# Perkecil citra yang terlalu besar
tinggi, lebar = bgr.shape[:2]
info_ukuran = f"{nama_citra} - {lebar} x {tinggi} px"
if max(tinggi, lebar) > MAKS_SISI:
    skala = MAKS_SISI / max(tinggi, lebar)
    bgr = cv2.resize(bgr, (int(lebar * skala), int(tinggi * skala)), interpolation=cv2.INTER_AREA)
    info_ukuran += f" (diperkecil menjadi {bgr.shape[1]} x {bgr.shape[0]} px)"
st.caption(info_ukuran)

rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)

# ----- Langkah 2: pilih operator -----
st.subheader("2. Pilih operator")

if "operator" not in st.session_state:
    st.session_state.operator = None

kolom = st.columns(len(OPERATOR))
for col, nama in zip(kolom, OPERATOR):
    with col:
        if st.button(nama, key=f"btn_{nama}", use_container_width=True):
            st.session_state.operator = nama

pilihan = st.session_state.operator

if pilihan is None:
    st.image(rgb, caption="Citra Asli", width=400)
    st.stop()

# ----- Hasil -----
warna = OPERATOR[pilihan]["warna"]
st.markdown(
    f"<div style='border-left:6px solid {warna};padding:.4rem .9rem;margin:1rem 0;"
    f"background:{warna}22;border-radius:6px'><b>{pilihan}</b> - {OPERATOR[pilihan]['info']}</div>",
    unsafe_allow_html=True,
)

with st.spinner(f"Memproses {pilihan}..."):
    if pilihan == "Sobel":
        tampil, hasil = proses_sobel(gray)
    elif pilihan == "Prewitt":
        tampil, hasil = proses_prewitt(gray)
    elif pilihan == "Roberts":
        tampil, hasil = proses_roberts(gray)
    else:
        tampil, hasil = proses_canny(gray, t1a, t2a, t1b, t2b)

daftar = [(rgb, "Citra Asli")] + tampil
kolom_hasil = st.columns(len(daftar))
for col, (img, judul) in zip(kolom_hasil, daftar):
    with col:
        st.image(img, caption=judul, use_container_width=True, clamp=True)

if pilihan == "Canny":
    m1, m2 = st.columns(2)
    # tampil[1] dan tampil[2] adalah dua hasil Canny
    m1.metric(f"Piksel tepi Canny ({t1a}, {t2a})", int(np.count_nonzero(tampil[1][0])))
    m2.metric(f"Piksel tepi Canny ({t1b}, {t2b})", int(np.count_nonzero(tampil[2][0])))

# Tombol unduh hasil
ok, png = cv2.imencode(".png", hasil)
if ok:
    st.download_button(f"Unduh hasil {pilihan} (PNG)", png.tobytes(),
                       file_name=f"hasil_{pilihan.lower()}.png", mime="image/png")
