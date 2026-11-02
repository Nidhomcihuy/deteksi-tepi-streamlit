from pathlib import Path

import cv2
import numpy as np
import streamlit as st

st.set_page_config(page_title="Segmentasi Hijau Tumbuhan", page_icon="🌿", layout="wide")

# Batas sisi terpanjang citra agar proses tetap cepat di server
MAKS_SISI = 1600

# ---------------------------------------------------------------
# Konfigurasi metode segmentasi (warna & deskripsi)
# ---------------------------------------------------------------
METODE = {
    "HSV": {
        "warna": "#16a34a",
        "info": "Menyaring rentang warna hijau (Hue) pada ruang warna HSV. "
                "Atur rentang di sidebar.",
    },
    "ExG": {
        "warna": "#65a30d",
        "info": "Excess Green Index (2G - R - B) + ambang Otsu. "
                "Menonjolkan dominasi hijau, tahan perubahan cahaya.",
    },
    "Lab": {
        "warna": "#0d9488",
        "info": "Kanal a* pada ruang warna CIELAB (hijau = nilai a* rendah) + ambang Otsu.",
    },
    "Gabungan": {
        "warna": "#047857",
        "info": "Irisan mask HSV dan ExG: hanya piksel yang lolos kedua metode. "
                "Hasil paling ketat, mengurangi salah deteksi.",
    },
}

# CSS: beri warna tiap tombol berdasarkan key (class .st-key-<key>)
css = ""
for nama, o in METODE.items():
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
# Fungsi segmentasi hijau
# ---------------------------------------------------------------
def mask_hsv(bgr, h_min, h_max, s_min, v_min):
    """Mask hijau dari rentang HSV. (OpenCV: H 0-179, S & V 0-255)."""
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    return cv2.inRange(hsv, (h_min, s_min, v_min), (h_max, 255, 255))


def mask_exg(bgr):
    """Excess Green: 2g - r - b pada koordinat kromatik, lalu ambang Otsu."""
    b, g, r = cv2.split(bgr.astype(np.float32))
    total = b + g + r + 1e-6
    exg = 2 * (g / total) - (r / total) - (b / total)
    exg_norm = cv2.normalize(exg, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    _, mask = cv2.threshold(exg_norm, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return mask, exg_norm


def mask_lab(bgr):
    """Kanal a* (hijau = rendah) dibalik, dihaluskan, lalu ambang Otsu."""
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    a_terbalik = 255 - lab[:, :, 1]
    a_halus = cv2.GaussianBlur(a_terbalik, (5, 5), 0)
    _, mask = cv2.threshold(a_halus, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return mask, a_halus


def bersihkan_mask(mask, ukuran):
    """Buka (hapus bintik kecil) lalu tutup (isi lubang kecil)."""
    if ukuran <= 0:
        return mask
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ukuran, ukuran))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)


def segmentasi(metode, bgr, p):
    """Mengembalikan (mask, daftar citra antara [(img, judul)])."""
    antara = []
    if metode == "HSV":
        mask = mask_hsv(bgr, p["h_min"], p["h_max"], p["s_min"], p["v_min"])
    elif metode == "ExG":
        mask, exg = mask_exg(bgr)
        antara.append((exg, "Peta Excess Green"))
    elif metode == "Lab":
        mask, a_img = mask_lab(bgr)
        antara.append((a_img, "Kanal a* (dibalik)"))
    else:  # Gabungan
        m_hsv = mask_hsv(bgr, p["h_min"], p["h_max"], p["s_min"], p["v_min"])
        m_exg, _ = mask_exg(bgr)
        mask = cv2.bitwise_and(m_hsv, m_exg)
    return bersihkan_mask(mask, p["morf"]), antara


def terapkan_mask(rgb, mask, warna_latar):
    """Pertahankan piksel hijau, ganti sisanya dengan warna latar."""
    hasil = np.full_like(rgb, warna_latar)
    hasil[mask > 0] = rgb[mask > 0]
    return hasil


def buat_overlay(rgb, mask):
    """Sorot area hijau dengan warna magenta semi transparan."""
    sorot = np.zeros_like(rgb)
    sorot[:] = (255, 0, 200)
    campur = cv2.addWeighted(rgb, 0.45, sorot, 0.55, 0)
    overlay = rgb.copy()
    overlay[mask > 0] = campur[mask > 0]
    return overlay


def decode_citra(file_obj):
    """Decode berkas unggahan/kamera menjadi array BGR (None bila gagal)."""
    data = np.frombuffer(file_obj.getvalue(), dtype=np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def path_contoh():
    path = Path(__file__).resolve().parent / "spider.jpeg"
    return path if path.exists() else None


LATAR = {"Hitam": (0, 0, 0), "Putih": (255, 255, 255), "Abu-abu": (128, 128, 128)}

# ---------------------------------------------------------------
# Sidebar (parameter)
# ---------------------------------------------------------------
with st.sidebar:
    st.header("Pengaturan")

    st.subheader("Rentang HSV hijau")
    st.caption("Dipakai oleh metode HSV dan Gabungan.")
    h_min, h_max = st.slider("Hue (warna)", 0, 179, (35, 85),
                             help="Hijau pada OpenCV berkisar sekitar 35-85.")
    s_min = st.slider("Saturasi minimum", 0, 255, 40,
                      help="Naikkan untuk membuang warna pucat/keabu-abuan.")
    v_min = st.slider("Kecerahan minimum (Value)", 0, 255, 40,
                      help="Naikkan untuk membuang area yang terlalu gelap.")

    st.subheader("Pembersihan mask")
    morf = st.slider("Ukuran kernel morfologi", 0, 15, 3,
                     help="0 = tanpa pembersihan. Semakin besar, bintik kecil makin hilang.")

    st.subheader("Tampilan")
    nama_latar = st.radio("Warna latar hasil", list(LATAR), horizontal=True)

params = {"h_min": h_min, "h_max": h_max, "s_min": s_min, "v_min": v_min, "morf": morf}

# ---------------------------------------------------------------
# Halaman utama
# ---------------------------------------------------------------
st.title("Segmentasi Hijau Tumbuhan")
st.caption("Unggah foto tumbuhan, lalu pilih metode untuk memisahkan area berwarna hijau.")

# ----- Langkah 1: sumber citra -----
st.subheader("1. Pilih citra")

sumber_opsi = ["Unggah gambar", "Ambil dari kamera"]
if path_contoh() is not None:
    sumber_opsi.append("Gunakan contoh (spider.jpeg)")

sumber = st.radio("Sumber citra", sumber_opsi, horizontal=True, label_visibility="collapsed")

bgr = None
nama_citra = ""
unggahan = None

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
    if unggahan is not None:
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

# ----- Langkah 2: pilih metode -----
st.subheader("2. Pilih metode segmentasi")

if "metode" not in st.session_state:
    st.session_state.metode = None

kolom = st.columns(len(METODE))
for col, nama in zip(kolom, METODE):
    with col:
        if st.button(nama, key=f"btn_{nama}", use_container_width=True):
            st.session_state.metode = nama

pilihan = st.session_state.metode

if pilihan is None:
    st.image(rgb, caption="Citra Asli", width=400)
    st.stop()

# ----- Hasil -----
warna = METODE[pilihan]["warna"]
st.markdown(
    f"<div style='border-left:6px solid {warna};padding:.4rem .9rem;margin:1rem 0;"
    f"background:{warna}22;border-radius:6px'><b>{pilihan}</b> - {METODE[pilihan]['info']}</div>",
    unsafe_allow_html=True,
)

with st.spinner(f"Memproses {pilihan}..."):
    mask, antara = segmentasi(pilihan, bgr, params)
    hasil = terapkan_mask(rgb, mask, LATAR[nama_latar])
    overlay = buat_overlay(rgb, mask)

daftar = [(rgb, "Citra Asli")] + antara + [
    (mask, "Mask Hijau"),
    (hasil, "Hasil Segmentasi"),
    (overlay, "Overlay Area Hijau"),
]
kolom_hasil = st.columns(len(daftar))
for col, (img, judul) in zip(kolom_hasil, daftar):
    with col:
        st.image(img, caption=judul, use_container_width=True, clamp=True)

# Statistik area hijau
piksel_hijau = int(np.count_nonzero(mask))
persen = piksel_hijau / mask.size * 100
m1, m2, m3 = st.columns(3)
m1.metric("Area hijau", f"{persen:.1f}%")
m2.metric("Piksel hijau", f"{piksel_hijau:,}")
m3.metric("Piksel non-hijau", f"{mask.size - piksel_hijau:,}")

# Tombol unduh
d1, d2 = st.columns(2)
ok1, png_hasil = cv2.imencode(".png", cv2.cvtColor(hasil, cv2.COLOR_RGB2BGR))
ok2, png_mask = cv2.imencode(".png", mask)
if ok1:
    d1.download_button(f"Unduh hasil segmentasi {pilihan} (PNG)", png_hasil.tobytes(),
                       file_name=f"segmentasi_{pilihan.lower()}.png", mime="image/png",
                       use_container_width=True)
if ok2:
    d2.download_button(f"Unduh mask {pilihan} (PNG)", png_mask.tobytes(),
                       file_name=f"mask_{pilihan.lower()}.png", mime="image/png",
                       use_container_width=True)
