import tempfile
from datetime import datetime
from io import BytesIO
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image, ImageOps

try:
    from openpyxl import Workbook
    from openpyxl.chart import BarChart, Reference
    from openpyxl.drawing.image import Image as XLImage
    from openpyxl.styles import Alignment, Font, PatternFill

    ADA_OPENPYXL = True
except ImportError:  # ekspor Excel dinonaktifkan bila openpyxl belum terpasang
    ADA_OPENPYXL = False

# =====================================================
# 1. KONFIGURASI APLIKASI
# =====================================================

st.set_page_config(
    page_title="Smart Plant Monitoring",
    page_icon="🌱",
    layout="wide"
)

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "outputsApp"
try:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
except OSError:  # folder proyek tidak bisa ditulis (mis. hosting read-only)
    OUTPUT_DIR = Path(tempfile.gettempdir()) / "outputsApp"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MASK_DIR = OUTPUT_DIR / "masks"
CSV_FILE = OUTPUT_DIR / "hasil_monitoring.csv"
MASK_DIR.mkdir(parents=True, exist_ok=True)

# Batas sisi terpanjang citra agar proses tetap cepat di server
MAKS_SISI = 1600

# Metode segmentasi. HSV adalah metode utama modul praktikum;
# tiga lainnya adalah metode pembanding tambahan.
METODE = {
    "HSV": "Metode modul: piksel yang berada di rentang H, S, V pada sidebar.",
    "ExG": "Pembanding: Excess Green (2G - R - B) + ambang Otsu. Tidak memakai slider HSV.",
    "Lab": "Pembanding: kanal a* CIELAB (hijau = a* rendah) + ambang Otsu. Tidak memakai slider HSV.",
    "Gabungan": "Pembanding: irisan mask HSV dan ExG (paling ketat).",
}


# =====================================================
# 2. FUNGSI SEGMENTASI CITRA
# =====================================================

def mask_exg(image_bgr):
    """Excess Green: 2g - r - b pada koordinat kromatik, lalu ambang Otsu."""
    b, g, r = cv2.split(image_bgr.astype(np.float32))
    total = b + g + r + 1e-6
    exg = 2 * (g / total) - (r / total) - (b / total)
    exg_norm = cv2.normalize(exg, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    _, mask = cv2.threshold(exg_norm, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return mask, exg_norm


def mask_lab(image_bgr):
    """Kanal a* (hijau = rendah) dibalik, dihaluskan, lalu ambang Otsu."""
    lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)
    a_terbalik = 255 - lab[:, :, 1]
    a_halus = cv2.GaussianBlur(a_terbalik, (5, 5), 0)
    _, mask = cv2.threshold(a_halus, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return mask, a_halus


def segmentasi_hijau(
    image_rgb,
    lower_hsv,
    upper_hsv,
    kernel_size=5,
    metode="HSV"
):
    """
    Melakukan segmentasi warna hijau (metode utama: HSV).

    Parameter:
    - image_rgb: gambar RGB
    - lower_hsv: batas bawah HSV
    - upper_hsv: batas atas HSV
    - kernel_size: ukuran kernel morfologi
    - metode: "HSV" (modul), atau pembanding "ExG", "Lab", "Gabungan"

    Return:
    - hsv, mask, segmented, total_pixels, green_pixels, percentage
    """

    # Konversi RGB ke HSV
    image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)

    # Membuat mask berdasarkan rentang HSV
    mask_hsv = cv2.inRange(
        hsv,
        np.array(lower_hsv, dtype=np.uint8),
        np.array(upper_hsv, dtype=np.uint8)
    )

    if metode == "ExG":
        mask = mask_exg(image_bgr)[0]
    elif metode == "Lab":
        mask = mask_lab(image_bgr)[0]
    elif metode == "Gabungan":
        mask = cv2.bitwise_and(mask_hsv, mask_exg(image_bgr)[0])
    else:
        mask = mask_hsv

    # Membuat kernel morfologi
    kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)

    # Menghilangkan noise kecil
    mask_open = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    # Menutup lubang kecil pada area tersegmentasi
    mask_clean = cv2.morphologyEx(mask_open, cv2.MORPH_CLOSE, kernel)

    # Mengambil piksel yang terdeteksi sebagai hijau
    green_pixels = cv2.countNonZero(mask_clean)

    # Jumlah seluruh piksel gambar
    total_pixels = mask_clean.size

    # Menghitung persentase piksel hijau
    percentage = (green_pixels / total_pixels) * 100 if total_pixels > 0 else 0

    # Menampilkan hanya bagian yang masuk mask
    segmented = cv2.bitwise_and(image_rgb, image_rgb, mask=mask_clean)

    return hsv, mask_clean, segmented, total_pixels, green_pixels, percentage


def peta_antara(metode, image_rgb):
    """Peta nilai sebelum ambang (hanya untuk ExG dan Lab)."""
    image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    if metode == "ExG":
        return mask_exg(image_bgr)[1], "Peta Excess Green"
    if metode == "Lab":
        return mask_lab(image_bgr)[1], "Kanal a* (dibalik)"
    return None


def buat_overlay(image_rgb, mask):
    """Sorot area hijau dengan warna magenta semi transparan."""
    sorot = np.zeros_like(image_rgb)
    sorot[:] = (255, 0, 200)
    campur = cv2.addWeighted(image_rgb, 0.45, sorot, 0.55, 0)
    overlay = image_rgb.copy()
    overlay[mask > 0] = campur[mask > 0]
    return overlay


# =====================================================
# 3. DATA UNTUK GRAFIK ANALISIS
# =====================================================

def hitung_perbandingan(image_rgb, lower_hsv, upper_hsv, kernel_size):
    """Persentase area hijau untuk semua metode (index = metode)."""
    baris = []
    for m in METODE:
        _, _, _, total, hijau, persen = segmentasi_hijau(
            image_rgb, lower_hsv, upper_hsv, kernel_size, m
        )
        baris.append({"Metode": m, "Piksel hijau": int(hijau),
                      "Total piksel": int(total),
                      "Area hijau (%)": round(persen, 2)})
    return pd.DataFrame(baris).set_index("Metode")


def hitung_histogram_hue(hsv, mask):
    """Histogram Hue (36 bin, lebar 5) untuk seluruh piksel dan piksel hijau."""
    hue = hsv[:, :, 0]
    semua, _ = np.histogram(hue, bins=36, range=(0, 180))
    hijau, _ = np.histogram(hue[mask > 0], bins=36, range=(0, 180))
    label = [f"{i * 5:03d}" for i in range(36)]  # nol di depan agar urutan benar
    return pd.DataFrame({"Semua piksel": semua, "Piksel hijau": hijau}, index=label)


def hitung_rgb_rata(image_rgb, mask):
    """Rata-rata kanal R, G, B pada seluruh citra dan pada area hijau."""
    seluruh = image_rgb.reshape(-1, 3).mean(axis=0)
    hijau = image_rgb[mask > 0].mean(axis=0) if np.any(mask > 0) else np.zeros(3)
    return pd.DataFrame({"Seluruh citra": seluruh.round(1), "Area hijau": hijau.round(1)},
                        index=["R", "G", "B"])


# =====================================================
# 4. FUNGSI MENYIMPAN HASIL KE CSV
# =====================================================

def simpan_hasil(data_baru):
    """
    Menyimpan satu hasil monitoring ke CSV.
    """

    df_baru = pd.DataFrame([data_baru])

    if CSV_FILE.exists():
        df_lama = pd.read_csv(CSV_FILE)

        df_gabungan = pd.concat(
            [df_lama, df_baru],
            ignore_index=True
        )
    else:
        df_gabungan = df_baru

    df_gabungan.to_csv(
        CSV_FILE,
        index=False,
        encoding="utf-8-sig"
    )

    return df_gabungan


# =====================================================
# 5. FUNGSI EKSPOR EXCEL (.xlsx)
# =====================================================

def _gaya_header(ws, baris, jumlah_kolom):
    isi = PatternFill("solid", start_color="16A34A")
    for c in range(1, jumlah_kolom + 1):
        sel = ws.cell(row=baris, column=c)
        sel.fill = isi
        sel.font = Font(bold=True, color="FFFFFF")
        sel.alignment = Alignment(horizontal="center")


def _lebar_kolom(ws, lebar):
    for huruf, w in lebar.items():
        ws.column_dimensions[huruf].width = w


def _grafik_bar(ws, judul, sumbu_y, kol_awal, kol_akhir, baris_akhir, anchor,
                lebar=16, tinggi=8, kol_kategori=1):
    grafik = BarChart()
    grafik.type = "col"
    grafik.title = judul
    grafik.y_axis.title = sumbu_y
    grafik.x_axis.delete = False
    grafik.y_axis.delete = False
    data = Reference(ws, min_col=kol_awal, max_col=kol_akhir, min_row=1, max_row=baris_akhir)
    kategori = Reference(ws, min_col=kol_kategori, min_row=2, max_row=baris_akhir)
    grafik.add_data(data, titles_from_data=True)
    grafik.set_categories(kategori)
    grafik.width, grafik.height = lebar, tinggi
    ws.add_chart(grafik, anchor)


def _tulis_riwayat(ws, df):
    """Tulis tabel riwayat monitoring ke sheet, lengkap dengan grafik persentase."""
    df = df.copy()
    if "persentase_hijau" in df.columns:
        df["persentase_hijau"] = pd.to_numeric(df["persentase_hijau"], errors="coerce")
    kolom = list(df.columns)
    ws.append(kolom)
    _gaya_header(ws, 1, len(kolom))
    for baris in df.astype(object).where(pd.notna(df), "").itertuples(index=False):
        ws.append([v.item() if hasattr(v, "item") else v for v in baris])
    for i, nama in enumerate(kolom, start=1):
        panjang = max([len(str(nama))] + [len(str(v)) for v in df[nama].head(50)])
        ws.column_dimensions[ws.cell(row=1, column=i).column_letter].width = min(panjang + 3, 40)
    if "persentase_hijau" in kolom and len(df) > 0:
        kol_p = kolom.index("persentase_hijau") + 1
        kol_id = kolom.index("id_tanaman") + 1 if "id_tanaman" in kolom else 1
        _grafik_bar(ws, "Persentase piksel tersegmentasi hijau", "Persentase (%)",
                    kol_p, kol_p, len(df) + 1, f"{ws.cell(row=1, column=len(kolom) + 2).column_letter}2",
                    lebar=20, tinggi=9, kol_kategori=kol_id)


def bangun_xlsx_riwayat(df_history):
    """Excel berisi riwayat monitoring saja."""
    wb = Workbook()
    ws = wb.active
    ws.title = "Riwayat Monitoring"
    _tulis_riwayat(ws, df_history)
    buf = BytesIO()
    wb.save(buf)
    return buf.getvalue()


def bangun_xlsx(info, perbandingan, hist_df, rgb_df, gambar, df_history=None):
    """Laporan Excel satu citra: ringkasan, tabel, grafik asli Excel, gambar, riwayat."""
    wb = Workbook()

    # ----- Ringkasan -----
    ws = wb.active
    ws.title = "Ringkasan"
    ws["A1"] = "Laporan Smart Plant Monitoring"
    ws["A1"].font = Font(bold=True, size=14)
    ws.append([])
    ws.append(["Parameter", "Nilai"])
    _gaya_header(ws, 3, 2)
    for baris in info:
        ws.append(list(baris))
    _lebar_kolom(ws, {"A": 30, "B": 32})

    # ----- Perbandingan metode -----
    ws2 = wb.create_sheet("Perbandingan Metode")
    ws2.append(["Metode", "Piksel hijau", "Total piksel", "Area hijau (%)", "Dipilih"])
    _gaya_header(ws2, 1, 5)
    metode_dipilih = dict(info).get("Metode terpilih")
    for metode, r in perbandingan.iterrows():
        ws2.append([metode, int(r["Piksel hijau"]), int(r["Total piksel"]),
                    float(r["Area hijau (%)"]), "Ya" if metode == metode_dipilih else ""])
    _lebar_kolom(ws2, {"A": 14, "B": 14, "C": 14, "D": 16, "E": 10})
    _grafik_bar(ws2, "Perbandingan area hijau antar metode", "Area hijau (%)",
                4, 4, len(perbandingan) + 1, "G2")

    # ----- Histogram Hue -----
    ws3 = wb.create_sheet("Histogram Hue")
    ws3.append(["Hue (mulai bin)", "Semua piksel", "Piksel hijau"])
    _gaya_header(ws3, 1, 3)
    for label, r in hist_df.iterrows():
        ws3.append([int(label), int(r["Semua piksel"]), int(r["Piksel hijau"])])
    _lebar_kolom(ws3, {"A": 16, "B": 14, "C": 14})
    _grafik_bar(ws3, "Histogram Hue", "Jumlah piksel", 2, 3, len(hist_df) + 1, "F2",
                lebar=24, tinggi=10)

    # ----- Rata-rata RGB -----
    ws4 = wb.create_sheet("Rata-rata RGB")
    ws4.append(["Kanal", "Seluruh citra", "Area hijau"])
    _gaya_header(ws4, 1, 3)
    for kanal, r in rgb_df.iterrows():
        ws4.append([kanal, float(r["Seluruh citra"]), float(r["Area hijau"])])
    _lebar_kolom(ws4, {"A": 10, "B": 16, "C": 14})
    _grafik_bar(ws4, "Rata-rata kanal RGB", "Nilai rata-rata (0-255)", 2, 3,
                len(rgb_df) + 1, "F2")

    # ----- Citra -----
    ws5 = wb.create_sheet("Citra")
    ws5["A1"] = "Citra hasil pengolahan"
    ws5["A1"].font = Font(bold=True, size=14)
    for (judul, img), sel in zip(gambar, ["A3", "L3", "A28", "L28"]):
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

    # ----- Riwayat (jika sudah ada data tersimpan) -----
    if df_history is not None and not df_history.empty:
        _tulis_riwayat(wb.create_sheet("Riwayat Monitoring"), df_history)

    buf = BytesIO()
    wb.save(buf)
    return buf.getvalue()


def png_bytes(array):
    """Array (RGB atau grayscale) menjadi byte PNG lewat PIL."""
    buffer = BytesIO()
    Image.fromarray(array).save(buffer, format="PNG")
    return buffer.getvalue()


def baca_gambar(file_obj):
    """Baca berkas unggahan/kamera menjadi array RGB (None bila gagal)."""
    try:
        gambar = Image.open(BytesIO(file_obj.getvalue()))
        gambar = ImageOps.exif_transpose(gambar).convert("RGB")
        return np.array(gambar)
    except Exception:
        return None


# =====================================================
# 6. JUDUL APLIKASI
# =====================================================

st.title("🌱 Smart Plant Monitoring")

st.write(
    """
    Aplikasi monitoring tanaman menggunakan
    **Computer Vision, OpenCV, dan segmentasi warna HSV**.

    Aplikasi menghitung jumlah dan persentase piksel
    yang terdeteksi berwarna hijau pada gambar.
    """
)

st.divider()

# =====================================================
# 7. SIDEBAR: PARAMETER DAN PENGATURAN
# =====================================================

st.sidebar.header("⚙️ Pengaturan Segmentasi")

st.sidebar.write(
    "Atur rentang warna HSV untuk mendeteksi warna hijau."
)

# Nilai default mengikuti eksperimen sebelumnya
h_min = st.sidebar.slider(
    "Hue minimum",
    min_value=0,
    max_value=179,
    value=25
)

h_max = st.sidebar.slider(
    "Hue maksimum",
    min_value=0,
    max_value=179,
    value=95
)

s_min = st.sidebar.slider(
    "Saturation minimum",
    min_value=0,
    max_value=255,
    value=35
)

s_max = st.sidebar.slider(
    "Saturation maksimum",
    min_value=0,
    max_value=255,
    value=255
)

v_min = st.sidebar.slider(
    "Value minimum",
    min_value=0,
    max_value=255,
    value=30
)

v_max = st.sidebar.slider(
    "Value maksimum",
    min_value=0,
    max_value=255,
    value=255
)

kernel_size = st.sidebar.select_slider(
    "Ukuran kernel morfologi",
    options=[3, 5, 7, 9],
    value=5
)

if h_min > h_max:
    st.sidebar.error(
        "Hue minimum tidak boleh lebih besar "
        "daripada Hue maksimum."
    )
    st.stop()

if s_min > s_max or v_min > v_max:
    st.sidebar.error(
        "Batas minimum tidak boleh lebih besar "
        "daripada batas maksimum."
    )
    st.stop()

lower_hsv = [h_min, s_min, v_min]
upper_hsv = [h_max, s_max, v_max]

st.sidebar.divider()
st.sidebar.subheader("Opsi tambahan")
metode = st.sidebar.selectbox(
    "Metode segmentasi",
    list(METODE),
    help="HSV adalah metode pada modul praktikum. Metode lain dipakai sebagai pembanding.",
)
st.sidebar.caption(METODE[metode])
perkecil = st.sidebar.checkbox(
    f"Perkecil citra besar (maks. {MAKS_SISI} px)",
    value=True,
    help="Mempercepat proses di server. Persentase hijau hampir tidak berubah.",
)

# =====================================================
# 8. FORM INPUT DATA TANAMAN
# =====================================================

st.subheader("1. Input Data Tanaman")

col_input1, col_input2 = st.columns(2)

with col_input1:
    plant_id = st.text_input(
        "ID Tanaman",
        value="P001",
        help="Contoh: P001, P002, atau P003."
    )

with col_input2:
    lokasi = st.text_input(
        "Lokasi Pengamatan",
        value="Kebun Percobaan"
    )

contoh = BASE_DIR / "spider.jpeg"
sumber_opsi = ["Unggah gambar", "Ambil dari kamera"]
if contoh.exists():
    sumber_opsi.append("Gunakan contoh (spider.jpeg)")
sumber = st.radio("Sumber citra", sumber_opsi, horizontal=True)

image_rgb = None
nama_citra = ""
uploaded_file = None

if sumber == "Unggah gambar":
    uploaded_file = st.file_uploader(
        "Unggah foto tanaman atau daun pepaya",
        type=["jpg", "jpeg", "png", "bmp"]
    )
    if uploaded_file is not None:
        image_rgb = baca_gambar(uploaded_file)
        nama_citra = uploaded_file.name
        if image_rgb is None:
            st.error("Gambar gagal dibaca. Pastikan file valid (JPG, JPEG, PNG, BMP).")
elif sumber == "Ambil dari kamera":
    foto = st.camera_input("Ambil foto tanaman")
    if foto is not None:
        image_rgb = baca_gambar(foto)
        nama_citra = "foto_kamera.png"
else:
    image_rgb = np.array(Image.open(contoh).convert("RGB"))
    nama_citra = contoh.name


# =====================================================
# 9. PEMROSESAN GAMBAR
# =====================================================

def tampilkan_hasil(image_rgb, nama_citra):
    tinggi, lebar = image_rgb.shape[:2]
    info_ukuran = f"{nama_citra} - {lebar} x {tinggi} px"
    if perkecil and max(tinggi, lebar) > MAKS_SISI:
        skala = MAKS_SISI / max(tinggi, lebar)
        image_rgb = cv2.resize(
            image_rgb, (int(lebar * skala), int(tinggi * skala)),
            interpolation=cv2.INTER_AREA,
        )
        info_ukuran += f" (diperkecil menjadi {image_rgb.shape[1]} x {image_rgb.shape[0]} px)"
    st.caption(info_ukuran)

    st.subheader("2. Hasil Pengolahan Citra")

    (
        hsv,
        mask,
        segmented,
        total_pixels,
        green_pixels,
        percentage
    ) = segmentasi_hijau(
        image_rgb,
        lower_hsv,
        upper_hsv,
        kernel_size,
        metode
    )

    # -------------------------------------------------
    # Menampilkan gambar
    # -------------------------------------------------

    overlay = buat_overlay(image_rgb, mask)
    antara = peta_antara(metode, image_rgb)

    daftar = [(image_rgb, "Gambar Asli")]
    if antara is not None:
        daftar.append(antara)
    daftar += [
        (mask, "Mask Segmentasi Hijau"),
        (segmented, "Hasil Segmentasi"),
        (overlay, "Overlay Area Hijau"),
    ]

    for col, (img, judul) in zip(st.columns(len(daftar)), daftar):
        with col:
            st.image(
                img,
                caption=judul,
                width="stretch"
            )

    # -------------------------------------------------
    # Menampilkan hasil perhitungan
    # -------------------------------------------------

    st.subheader("3. Hasil Perhitungan")

    metric1, metric2, metric3 = st.columns(3)

    metric1.metric(
        "Total Piksel",
        f"{total_pixels:,}"
    )

    metric2.metric(
        "Piksel Hijau",
        f"{green_pixels:,}"
    )

    metric3.metric(
        "Persentase Hijau",
        f"{percentage:.2f}%"
    )

    st.progress(
        min(max(percentage / 100, 0.0), 1.0)
    )

    st.info(
        "Persentase ini menunjukkan proporsi piksel "
        "yang lolos segmentasi HSV, bukan otomatis "
        "persentase luas daun atau tingkat kesehatan tanaman."
    )

    # -------------------------------------------------
    # Grafik analisis citra
    # -------------------------------------------------

    st.markdown("#### Grafik analisis citra")

    perbandingan = hitung_perbandingan(image_rgb, lower_hsv, upper_hsv, kernel_size)
    hist_df = hitung_histogram_hue(hsv, mask)
    rgb_df = hitung_rgb_rata(image_rgb, mask)

    g1, g2, g3 = st.columns(3)
    with g1:
        st.markdown("**Perbandingan area hijau antar metode (%)**")
        st.bar_chart(perbandingan[["Area hijau (%)"]])
    with g2:
        st.markdown("**Histogram Hue**")
        st.bar_chart(hist_df)
        st.caption("Sumbu X: awal rentang Hue (0-179). Hijau berada sekitar 35-85.")
    with g3:
        st.markdown("**Rata-rata kanal RGB**")
        st.bar_chart(rgb_df)

    # -------------------------------------------------
    # Data hasil monitoring
    # -------------------------------------------------

    safe_plant_id = "".join(
        c for c in plant_id.strip()
        if c.isalnum() or c in ("-", "_")
    )

    if not safe_plant_id:
        safe_plant_id = "tanaman"

    safe_filename = "".join(
        c for c in Path(nama_citra).stem
        if c.isalnum() or c in ("-", "_")
    ) or "gambar"

    data_monitoring = {
        "waktu_pemrosesan": datetime.now().strftime(
            "%Y-%m-%d %H:%M:%S"
        ),
        "id_tanaman": plant_id.strip(),
        "lokasi": lokasi.strip(),
        "nama_file": nama_citra,
        "total_piksel": total_pixels,
        "piksel_hijau": green_pixels,
        "persentase_hijau": round(percentage, 2),
        "hsv_bawah": str(lower_hsv),
        "hsv_atas": str(upper_hsv),
        "ukuran_kernel": kernel_size,
        "metode": metode,
    }

    # Penanda hasil ini (berubah bila gambar/parameter/ID berubah)
    tanda_hasil = (
        nama_citra, total_pixels, green_pixels, plant_id.strip(),
        lokasi.strip(), tuple(lower_hsv), tuple(upper_hsv), kernel_size, metode,
    )

    # -------------------------------------------------
    # Tombol penyimpanan
    # -------------------------------------------------

    st.markdown("#### Simpan dan unduh")

    if st.button(
        "💾 Simpan Hasil Monitoring",
        type="primary"
    ):

        # Mencegah satu hasil yang sama disimpan berulang.
        # Menyimpan lagi diperbolehkan setelah gambar,
        # parameter, atau data tanaman berubah.
        if st.session_state.get("hasil_tersimpan") == tanda_hasil:
            st.warning(
                "Hasil ini sudah disimpan. "
                "Ubah gambar atau parameter untuk pengukuran berikutnya."
            )
        else:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            mask_path = MASK_DIR / f"{safe_plant_id}_{safe_filename}_{timestamp}_mask.png"
            cv2.imwrite(str(mask_path), mask)

            try:
                data_monitoring["file_mask"] = str(mask_path.relative_to(BASE_DIR))
            except ValueError:
                data_monitoring["file_mask"] = str(mask_path)

            simpan_hasil(data_monitoring)
            st.session_state["hasil_tersimpan"] = tanda_hasil

            st.success(
                f"Hasil monitoring tanaman "
                f"{plant_id} berhasil disimpan."
            )

    # -------------------------------------------------
    # Tombol unduh
    # -------------------------------------------------

    d1, d2, d3 = st.columns(3)

    d1.download_button(
        label="⬇️ Unduh Mask Segmentasi",
        data=png_bytes(mask),
        file_name=f"{safe_plant_id}_mask.png",
        mime="image/png",
        width="stretch",
    )

    d2.download_button(
        label="⬇️ Unduh Hasil Segmentasi",
        data=png_bytes(segmented),
        file_name=f"{safe_plant_id}_segmentasi.png",
        mime="image/png",
        width="stretch",
    )

    if ADA_OPENPYXL:
        df_riwayat = pd.read_csv(CSV_FILE) if CSV_FILE.exists() else None
        info = [
            ("ID tanaman", plant_id.strip()),
            ("Lokasi pengamatan", lokasi.strip()),
            ("File citra", nama_citra),
            ("Ukuran diproses (px)", f"{image_rgb.shape[1]} x {image_rgb.shape[0]}"),
            ("Metode terpilih", metode),
            ("Total piksel", int(total_pixels)),
            ("Piksel hijau", int(green_pixels)),
            ("Persentase hijau (%)", round(float(percentage), 2)),
            ("HSV bawah [H, S, V]", str(lower_hsv)),
            ("HSV atas [H, S, V]", str(upper_hsv)),
            ("Ukuran kernel", int(kernel_size)),
            ("Waktu pemrosesan", data_monitoring["waktu_pemrosesan"]),
        ]
        d3.download_button(
            label="📊 Unduh Laporan Excel (.xlsx)",
            data=bangun_xlsx(
                info, perbandingan, hist_df, rgb_df,
                [("Gambar Asli", image_rgb), ("Mask Hijau", mask),
                 ("Hasil Segmentasi", segmented), ("Overlay", overlay)],
                df_riwayat,
            ),
            file_name=f"laporan_{safe_plant_id}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            width="stretch",
        )
    else:
        d3.warning("Ekspor Excel butuh openpyxl. Tambahkan `openpyxl` ke requirements.txt.")


if image_rgb is not None:
    tampilkan_hasil(image_rgb, nama_citra)
elif uploaded_file is None:
    st.info("Belum ada citra. Unggah gambar, ambil foto, atau pilih contoh untuk memulai.")

# =====================================================
# 10. RIWAYAT MONITORING
# =====================================================

st.divider()

st.subheader("4. Riwayat Monitoring")

if CSV_FILE.exists():

    df_history = pd.read_csv(CSV_FILE)

    if not df_history.empty:

        st.dataframe(
            df_history,
            width="stretch"
        )

        h1, h2 = st.columns(2)

        h1.download_button(
            label="⬇️ Unduh Riwayat CSV",
            data=df_history.to_csv(
                index=False
            ).encode("utf-8-sig"),
            file_name="hasil_monitoring.csv",
            mime="text/csv",
            width="stretch",
        )

        if ADA_OPENPYXL:
            h2.download_button(
                label="📊 Unduh Riwayat Excel (.xlsx)",
                data=bangun_xlsx_riwayat(df_history),
                file_name="hasil_monitoring.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                width="stretch",
            )

        # Grafik persentase piksel hijau
        st.subheader("5. Grafik Monitoring")

        if (
            "id_tanaman" in df_history.columns
            and "persentase_hijau" in df_history.columns
        ):

            df_plot = df_history.copy()

            df_plot["persentase_hijau"] = pd.to_numeric(
                df_plot["persentase_hijau"],
                errors="coerce"
            )

            df_plot = df_plot.dropna(
                subset=["persentase_hijau"]
            )

            if not df_plot.empty:

                df_plot["label_grafik"] = (
                    df_plot["id_tanaman"].astype(str)
                    + " - "
                    + df_plot["nama_file"].astype(str)
                )

                fig, ax = plt.subplots(figsize=(12, 5))

                ax.bar(
                    df_plot["label_grafik"],
                    df_plot["persentase_hijau"]
                )

                ax.set_title(
                    "Persentase Piksel Tersegmentasi Hijau"
                )

                ax.set_xlabel(
                    "Tanaman dan Nama File"
                )

                ax.set_ylabel(
                    "Persentase (%)"
                )

                ax.tick_params(
                    axis="x",
                    labelrotation=45
                )

                ax.grid(
                    axis="y",
                    linestyle="--",
                    alpha=0.4
                )

                fig.tight_layout()

                st.pyplot(fig)

                # Menyimpan grafik ke folder outputs
                grafik_path = (
                    OUTPUT_DIR /
                    "grafik_persentase_hijau.png"
                )

                fig.savefig(
                    grafik_path,
                    dpi=300,
                    bbox_inches="tight"
                )

                grafik_buffer = BytesIO()
                fig.savefig(grafik_buffer, format="png", dpi=300, bbox_inches="tight")

                plt.close(fig)

                st.download_button(
                    label="⬇️ Unduh Grafik Monitoring (PNG)",
                    data=grafik_buffer.getvalue(),
                    file_name="grafik_persentase_hijau.png",
                    mime="image/png",
                )

    else:
        st.info("Belum ada data monitoring.")

else:
    st.info(
        "Belum ada riwayat monitoring. "
        "Unggah gambar dan simpan hasil pertama Anda."
    )

# =====================================================
# 11. INFORMASI APLIKASI
# =====================================================

st.divider()

st.caption(
    "Smart Plant Monitoring | "
    "Python, OpenCV, HSV, Matplotlib, dan Streamlit"
)
