from pathlib import Path

import cv2
import numpy as np
import pandas as pd
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
# Data untuk grafik & laporan Excel
# ---------------------------------------------------------------
def hitung_perbandingan(bgr, p):
    """Persentase area hijau untuk semua metode (DataFrame, index = metode)."""
    total = bgr.shape[0] * bgr.shape[1]
    baris = []
    for m in METODE:
        mk, _ = segmentasi(m, bgr, p)
        n = int(np.count_nonzero(mk))
        baris.append({"Metode": m, "Piksel hijau": n, "Total piksel": total,
                      "Area hijau (%)": round(n / total * 100, 2)})
    return pd.DataFrame(baris).set_index("Metode")


def hitung_histogram_hue(bgr, mask):
    """Histogram Hue (36 bin, lebar 5) untuk seluruh piksel dan piksel hijau."""
    hue = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)[:, :, 0]
    semua, _ = np.histogram(hue, bins=36, range=(0, 180))
    hijau, _ = np.histogram(hue[mask > 0], bins=36, range=(0, 180))
    label = [f"{i * 5:03d}" for i in range(36)]  # nol di depan agar urutan benar
    return pd.DataFrame({"Semua piksel": semua, "Piksel hijau": hijau}, index=label)


def hitung_rgb_rata(rgb, mask):
    """Rata-rata kanal R, G, B pada seluruh citra dan pada area hijau."""
    seluruh = rgb.reshape(-1, 3).mean(axis=0)
    hijau = rgb[mask > 0].mean(axis=0) if np.any(mask > 0) else np.zeros(3)
    return pd.DataFrame({"Seluruh citra": seluruh.round(1), "Area hijau": hijau.round(1)},
                        index=["R", "G", "B"])


def bangun_xlsx(nama_citra, ukuran_px, pilihan, params, nama_latar,
                perbandingan, hist_df, rgb_df, gambar):
    """Susun laporan Excel (ringkasan, tabel, grafik asli Excel, dan gambar)."""
    from io import BytesIO

    from openpyxl import Workbook
    from openpyxl.chart import BarChart, Reference
    from openpyxl.drawing.image import Image as XLImage
    from openpyxl.styles import Alignment, Font, PatternFill

    isi_header = PatternFill("solid", start_color="16A34A")
    font_header = Font(bold=True, color="FFFFFF")

    def gaya_header(ws, baris, jumlah_kolom):
        for c in range(1, jumlah_kolom + 1):
            sel = ws.cell(row=baris, column=c)
            sel.fill = isi_header
            sel.font = font_header
            sel.alignment = Alignment(horizontal="center")

    def lebar_kolom(ws, lebar):
        for huruf, w in lebar.items():
            ws.column_dimensions[huruf].width = w

    def buat_bar(ws, judul, sumbu_y, kol_data_awal, kol_data_akhir, baris_akhir,
                 anchor, lebar=16, tinggi=8):
        grafik = BarChart()
        grafik.type = "col"
        grafik.title = judul
        grafik.y_axis.title = sumbu_y
        grafik.x_axis.delete = False
        grafik.y_axis.delete = False
        data = Reference(ws, min_col=kol_data_awal, max_col=kol_data_akhir,
                         min_row=1, max_row=baris_akhir)
        kategori = Reference(ws, min_col=1, min_row=2, max_row=baris_akhir)
        grafik.add_data(data, titles_from_data=True)
        grafik.set_categories(kategori)
        grafik.width, grafik.height = lebar, tinggi
        ws.add_chart(grafik, anchor)

    wb = Workbook()

    # ----- Sheet 1: Ringkasan -----
    ws = wb.active
    ws.title = "Ringkasan"
    ws["A1"] = "Laporan Segmentasi Hijau Tumbuhan"
    ws["A1"].font = Font(bold=True, size=14)
    ws.append([])
    ws.append(["Parameter", "Nilai"])
    gaya_header(ws, 3, 2)
    terpilih = perbandingan.loc[pilihan]
    for baris in [
        ("File citra", nama_citra),
        ("Ukuran diproses (px)", ukuran_px),
        ("Metode terpilih", pilihan),
        ("Area hijau (%)", float(terpilih["Area hijau (%)"])),
        ("Piksel hijau", int(terpilih["Piksel hijau"])),
        ("Total piksel", int(terpilih["Total piksel"])),
        ("Hue minimum", int(params["h_min"])),
        ("Hue maksimum", int(params["h_max"])),
        ("Saturasi minimum", int(params["s_min"])),
        ("Kecerahan minimum (Value)", int(params["v_min"])),
        ("Kernel morfologi", int(params["morf"])),
        ("Warna latar hasil", nama_latar),
    ]:
        ws.append(list(baris))
    lebar_kolom(ws, {"A": 30, "B": 28})

    # ----- Sheet 2: Perbandingan metode -----
    ws2 = wb.create_sheet("Perbandingan Metode")
    ws2.append(["Metode", "Piksel hijau", "Total piksel", "Area hijau (%)", "Dipilih"])
    gaya_header(ws2, 1, 5)
    for metode, r in perbandingan.iterrows():
        ws2.append([metode, int(r["Piksel hijau"]), int(r["Total piksel"]),
                    float(r["Area hijau (%)"]), "Ya" if metode == pilihan else ""])
    lebar_kolom(ws2, {"A": 14, "B": 14, "C": 14, "D": 16, "E": 10})
    buat_bar(ws2, "Perbandingan area hijau antar metode", "Area hijau (%)",
             4, 4, len(perbandingan) + 1, "G2")

    # ----- Sheet 3: Histogram Hue -----
    ws3 = wb.create_sheet("Histogram Hue")
    ws3.append(["Hue (mulai bin)", "Semua piksel", "Piksel hijau"])
    gaya_header(ws3, 1, 3)
    for label, r in hist_df.iterrows():
        ws3.append([int(label), int(r["Semua piksel"]), int(r["Piksel hijau"])])
    lebar_kolom(ws3, {"A": 16, "B": 14, "C": 14})
    buat_bar(ws3, f"Histogram Hue - {pilihan}", "Jumlah piksel",
             2, 3, len(hist_df) + 1, "F2", lebar=24, tinggi=10)

    # ----- Sheet 4: Rata-rata RGB -----
    ws4 = wb.create_sheet("Rata-rata RGB")
    ws4.append(["Kanal", "Seluruh citra", "Area hijau"])
    gaya_header(ws4, 1, 3)
    for kanal, r in rgb_df.iterrows():
        ws4.append([kanal, float(r["Seluruh citra"]), float(r["Area hijau"])])
    lebar_kolom(ws4, {"A": 10, "B": 16, "C": 14})
    buat_bar(ws4, "Rata-rata kanal RGB", "Nilai rata-rata (0-255)",
             2, 3, len(rgb_df) + 1, "F2")

    # ----- Sheet 5: Citra -----
    ws5 = wb.create_sheet("Citra")
    ws5["A1"] = "Citra hasil pengolahan"
    ws5["A1"].font = Font(bold=True, size=14)
    posisi = ["A3", "L3", "A28", "L28"]
    for (judul, img), sel in zip(gambar, posisi):
        sisi = max(img.shape[:2])
        if sisi > 360:
            skala = 360 / sisi
            img = cv2.resize(img, (int(img.shape[1] * skala), int(img.shape[0] * skala)),
                             interpolation=cv2.INTER_AREA)
        if img.ndim == 3:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        ok, png = cv2.imencode(".png", img)
        if not ok:
            continue
        kolom, baris = sel[0], int(sel[1:])
        ws5[f"{kolom}{baris - 1}"] = judul
        ws5[f"{kolom}{baris - 1}"].font = Font(bold=True)
        ws5.add_image(XLImage(BytesIO(png.tobytes())), sel)

    buf = BytesIO()
    wb.save(buf)
    return buf.getvalue()

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

# ----- Langkah 3: grafik analisis -----
st.subheader("3. Grafik analisis")

perbandingan = hitung_perbandingan(bgr, params)
hist_df = hitung_histogram_hue(bgr, mask)
rgb_df = hitung_rgb_rata(rgb, mask)

g1, g2, g3 = st.columns(3)
with g1:
    st.markdown("**Perbandingan area hijau antar metode (%)**")
    st.bar_chart(perbandingan[["Area hijau (%)"]])
with g2:
    st.markdown(f"**Histogram Hue - {pilihan}**")
    st.bar_chart(hist_df)
    st.caption("Sumbu X: awal rentang Hue (0-179). Hijau berada sekitar 35-85.")
with g3:
    st.markdown("**Rata-rata kanal RGB**")
    st.bar_chart(rgb_df)

with st.expander("Lihat tabel perbandingan metode"):
    st.dataframe(perbandingan, use_container_width=True)

# ----- Langkah 4: ekspor -----
st.subheader("4. Ekspor hasil")

d1, d2, d3 = st.columns(3)
ok1, png_hasil = cv2.imencode(".png", cv2.cvtColor(hasil, cv2.COLOR_RGB2BGR))
ok2, png_mask = cv2.imencode(".png", mask)
if ok1:
    d1.download_button(f"Hasil segmentasi {pilihan} (PNG)", png_hasil.tobytes(),
                       file_name=f"segmentasi_{pilihan.lower()}.png", mime="image/png",
                       use_container_width=True)
if ok2:
    d2.download_button(f"Mask {pilihan} (PNG)", png_mask.tobytes(),
                       file_name=f"mask_{pilihan.lower()}.png", mime="image/png",
                       use_container_width=True)

try:
    data_xlsx = bangun_xlsx(
        nama_citra, f"{bgr.shape[1]} x {bgr.shape[0]}", pilihan, params, nama_latar,
        perbandingan, hist_df, rgb_df,
        [("Citra Asli", rgb), ("Mask Hijau", mask),
         ("Hasil Segmentasi", hasil), ("Overlay", overlay)],
    )
    d3.download_button(f"Laporan Excel {pilihan} (.xlsx)", data_xlsx,
                       file_name=f"laporan_segmentasi_{pilihan.lower()}.xlsx",
                       mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                       use_container_width=True, type="primary")
except ImportError:
    d3.warning("Ekspor Excel butuh library openpyxl. Tambahkan `openpyxl` ke requirements.txt.")
