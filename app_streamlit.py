import base64
import sys
import tkinter as tk
from pathlib import Path

import cv2
import numpy as np
import matplotlib.pyplot as plt

# ---------------------------------------------------------------
# 1. Membaca citra
# ---------------------------------------------------------------
folder = Path(__file__).resolve().parent
nama_file = sys.argv[1] if len(sys.argv) > 1 else "spider.jpeg"
path_gambar = Path(nama_file)
if not path_gambar.is_absolute():
    path_gambar = folder / path_gambar

print("Lokasi gambar :", path_gambar)
print("File tersedia :", path_gambar.exists())

citra = cv2.imread(str(path_gambar))  # hasil baca: BGR
if citra is None:
    raise SystemExit("Gambar gagal dibaca! Pastikan file ada dan formatnya valid.")
print("Citra berhasil dibaca, ukuran:", citra.shape)

# BGR -> RGB (untuk ditampilkan Matplotlib) lalu RGB -> grayscale (1 kanal)
citra_asli = cv2.cvtColor(citra, cv2.COLOR_BGR2RGB)
gray = cv2.cvtColor(citra_asli, cv2.COLOR_RGB2GRAY)


# ---------------------------------------------------------------
# Fungsi bantu
# ---------------------------------------------------------------
def magnitudo_ke_uint8(gx, gy):
    """Magnitudo gradien sqrt(Gx^2 + Gy^2), dipotong ke rentang 0-255."""
    mag = cv2.magnitude(gx, gy)
    return np.clip(mag, 0, 255).astype(np.uint8)


def tampilkan(gambar_judul, baris, kolom, ukuran, nama_file_simpan):
    """Menampilkan beberapa gambar dalam satu figure dan menyimpannya."""
    plt.figure(figsize=ukuran)
    for i, (img, judul) in enumerate(gambar_judul, start=1):
        plt.subplot(baris, kolom, i)
        if img.ndim == 2:
            plt.imshow(img, cmap="gray", vmin=0, vmax=255)
        else:
            plt.imshow(img)
        plt.title(judul)
        plt.axis("off")
    plt.tight_layout()
    plt.savefig(folder / nama_file_simpan, dpi=150)
    print("Disimpan:", nama_file_simpan)


# ---------------------------------------------------------------
# 2. Operator deteksi tepi (masing-masing dibungkus dalam fungsi)
# ---------------------------------------------------------------
def proses_sobel():
    # CV_32F dipakai agar gradien negatif tidak terpotong
    sobel_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)

    sobel_gx = cv2.convertScaleAbs(sobel_x)
    sobel_gy = cv2.convertScaleAbs(sobel_y)
    sobel = magnitudo_ke_uint8(sobel_x, sobel_y)

    tampilkan(
        [(citra_asli, "Citra Asli"), (sobel_gx, "Gradient X"),
         (sobel_gy, "Gradient Y"), (sobel, "Sobel")],
        2, 2, (10, 8), "hasil_sobel.png",
    )


def proses_prewitt():
    kernel_x = np.array([[-1, 0, 1],
                         [-1, 0, 1],
                         [-1, 0, 1]], dtype=np.float32)
    kernel_y = np.array([[-1, -1, -1],
                         [0, 0, 0],
                         [1, 1, 1]], dtype=np.float32)

    prewitt_x = cv2.filter2D(gray, cv2.CV_32F, kernel_x)
    prewitt_y = cv2.filter2D(gray, cv2.CV_32F, kernel_y)

    prewitt_gx = cv2.convertScaleAbs(prewitt_x)
    prewitt_gy = cv2.convertScaleAbs(prewitt_y)
    prewitt = magnitudo_ke_uint8(prewitt_x, prewitt_y)

    tampilkan(
        [(citra_asli, "Citra Asli"), (prewitt_gx, "Gradient X"),
         (prewitt_gy, "Gradient Y"), (prewitt, "Prewitt")],
        2, 2, (10, 8), "hasil_prewitt.png",
    )


def proses_roberts():
    kernel_x = np.array([[1, 0],
                         [0, -1]], dtype=np.float32)
    kernel_y = np.array([[0, 1],
                         [-1, 0]], dtype=np.float32)

    roberts_x = cv2.filter2D(gray, cv2.CV_32F, kernel_x)
    roberts_y = cv2.filter2D(gray, cv2.CV_32F, kernel_y)

    roberts_gx = cv2.convertScaleAbs(roberts_x)
    roberts_gy = cv2.convertScaleAbs(roberts_y)
    roberts = magnitudo_ke_uint8(roberts_x, roberts_y)

    tampilkan(
        [(citra_asli, "Citra Asli"), (roberts_gx, "Gradient X (diagonal)"),
         (roberts_gy, "Gradient Y (diagonal)"), (roberts, "Roberts")],
        2, 2, (10, 8), "hasil_roberts.png",
    )


def proses_canny():
    # Haluskan dengan Gaussian blur untuk mengurangi noise
    blur = cv2.GaussianBlur(gray, (5, 5), sigmaX=1.4)

    canny_rendah = cv2.Canny(blur, threshold1=50, threshold2=100)   # lebih banyak tepi
    canny_tinggi = cv2.Canny(blur, threshold1=100, threshold2=200)  # hanya tepi kuat

    tampilkan(
        [(citra_asli, "Citra Asli"), (blur, "Grayscale + Gaussian Blur"),
         (canny_rendah, "Canny (50, 100)"), (canny_tinggi, "Canny (100, 200)")],
        2, 2, (10, 8), "hasil_canny.png",
    )

    print("Jumlah piksel tepi Canny (50, 100)  :", int(np.count_nonzero(canny_rendah)))
    print("Jumlah piksel tepi Canny (100, 200) :", int(np.count_nonzero(canny_tinggi)))


# Konfigurasi tiap operator: fungsi, warna aksen, ikon, dan deskripsi
OPERATOR = {
    "Sobel": {
        "fungsi": proses_sobel, "warna": "#3b82f6", "ikon": "S", "kunci": "1",
        "deskripsi": "Kernel 3x3, bobot 2 di tengah",
        "detail": "Tepi halus & tahan noise. Cocok sebagai pilihan umum.",
    },
    "Prewitt": {
        "fungsi": proses_prewitt, "warna": "#10b981", "ikon": "P", "kunci": "2",
        "deskripsi": "Kernel 3x3, bobot sama rata",
        "detail": "Mirip Sobel, perhitungan lebih sederhana.",
    },
    "Roberts": {
        "fungsi": proses_roberts, "warna": "#f59e0b", "ikon": "R", "kunci": "3",
        "deskripsi": "Kernel 2x2, selisih diagonal",
        "detail": "Cepat & tajam, tetapi lebih sensitif terhadap noise.",
    },
    "Canny": {
        "fungsi": proses_canny, "warna": "#ec4899", "ikon": "C", "kunci": "4",
        "deskripsi": "Blur + NMS + hysteresis threshold",
        "detail": "Tepi tipis dan bersih. Dibandingkan 2 pasang threshold.",
    },
}


# ---------------------------------------------------------------
# 3. Menu UI (tombol kartu berwarna dengan efek hover)
# ---------------------------------------------------------------
BG = "#11131c"          # latar jendela
BG_HEADER = "#191c2a"   # latar header
CARD = "#1f2335"        # latar kartu
CARD_HOVER = "#2b3150"  # latar kartu saat di-hover
TEKS = "#e6e9f5"
TEKS_REDUP = "#8a90ad"
FONT = "Segoe UI"


def menu_pilihan():
    """Tampilkan menu tombol. Mengembalikan nama operator, atau None jika keluar."""
    pilihan = {"nama": None}

    root = tk.Tk()
    root.title("Deteksi Tepi Citra")
    root.configure(bg=BG)
    root.resizable(False, False)

    # ----- Header -----
    header = tk.Frame(root, bg=BG_HEADER)
    header.pack(fill="x")

    tk.Label(header, text="DETEKSI TEPI CITRA", bg=BG_HEADER, fg=TEKS,
             font=(FONT, 16, "bold")).pack(pady=(18, 0))
    tk.Label(header, text="Pilih operator untuk memproses gambar", bg=BG_HEADER,
             fg=TEKS_REDUP, font=(FONT, 10)).pack(pady=(2, 10))

    # Thumbnail citra
    tinggi, lebar = citra.shape[:2]
    skala = min(380 / lebar, 140 / tinggi, 1.0)
    thumb = cv2.resize(citra, (max(1, int(lebar * skala)), max(1, int(tinggi * skala))))
    _, buf = cv2.imencode(".png", thumb)
    root.foto = tk.PhotoImage(data=base64.b64encode(buf.tobytes()))
    bingkai = tk.Frame(header, bg="#3a4066", padx=2, pady=2)
    bingkai.pack()
    tk.Label(bingkai, image=root.foto, bd=0).pack()

    tk.Label(header, text=f"{path_gambar.name}  -  {lebar} x {tinggi} px",
             bg=BG_HEADER, fg=TEKS_REDUP, font=(FONT, 9)).pack(pady=(8, 16))

    # Garis aksen berwarna di bawah header (gabungan warna semua operator)
    garis = tk.Frame(root, bg=BG, height=4)
    garis.pack(fill="x")
    for info in OPERATOR.values():
        tk.Frame(garis, bg=info["warna"], height=4).pack(side="left", fill="x", expand=True)

    # ----- Daftar kartu tombol -----
    area = tk.Frame(root, bg=BG)
    area.pack(padx=24, pady=(16, 0))

    status = tk.Label(root, text="Arahkan kursor ke salah satu operator",
                      bg=BG, fg=TEKS_REDUP, font=(FONT, 9, "italic"),
                      height=2, wraplength=380)

    def pilih(nama):
        pilihan["nama"] = nama
        root.destroy()

    def klik(nama):
        warna = OPERATOR[nama]["warna"]
        status.config(text=f"Memproses {nama}...", fg=warna)
        root.update_idletasks()
        root.after(150, lambda: pilih(nama))

    def buat_kartu(nama, info):
        warna = info["warna"]

        kartu = tk.Frame(area, bg=CARD, cursor="hand2")
        kartu.pack(fill="x", pady=5)

        stripe = tk.Frame(kartu, bg=warna, width=6)
        stripe.pack(side="left", fill="y")

        badge = tk.Label(kartu, text=info["ikon"], bg=warna, fg="white",
                         font=(FONT, 15, "bold"), width=3)
        badge.pack(side="left", padx=(14, 12), pady=12)

        teks = tk.Frame(kartu, bg=CARD)
        teks.pack(side="left", fill="x", expand=True)
        judul = tk.Label(teks, text=nama, bg=CARD, fg=TEKS, anchor="w",
                         font=(FONT, 13, "bold"))
        judul.pack(fill="x")
        deskripsi = tk.Label(teks, text=info["deskripsi"], bg=CARD, fg=TEKS_REDUP,
                             anchor="w", font=(FONT, 9))
        deskripsi.pack(fill="x")

        kunci = tk.Label(kartu, text=f"[{info['kunci']}]", bg=CARD, fg=TEKS_REDUP,
                         font=(FONT, 9))
        kunci.pack(side="right", padx=(0, 6))
        panah = tk.Label(kartu, text="\u203A", bg=CARD, fg=TEKS_REDUP,
                         font=(FONT, 20, "bold"))
        panah.pack(side="right", padx=(0, 4))

        yang_diwarnai = [kartu, teks, judul, deskripsi, kunci, panah]
        semua = yang_diwarnai + [stripe, badge]

        def masuk(_=None):
            for w in yang_diwarnai:
                w.config(bg=CARD_HOVER)
            judul.config(fg=warna)
            panah.config(fg=warna)
            stripe.config(width=12)
            status.config(text=info["detail"], fg=warna)

        def keluar(_=None):
            for w in yang_diwarnai:
                w.config(bg=CARD)
            judul.config(fg=TEKS)
            panah.config(fg=TEKS_REDUP)
            stripe.config(width=6)
            status.config(text="Arahkan kursor ke salah satu operator", fg=TEKS_REDUP)

        def tekan(_=None):
            for w in yang_diwarnai:
                w.config(bg=warna)
            judul.config(fg="white")
            deskripsi.config(fg="white")
            panah.config(fg="white")
            kunci.config(fg="white")
            klik(nama)

        for w in semua:
            w.bind("<Enter>", masuk)
            w.bind("<Leave>", keluar)
            w.bind("<Button-1>", tekan)

        # Shortcut keyboard
        root.bind(f"<Key-{info['kunci']}>", lambda e, n=nama: (masuk(), tekan()))

    for nama, info in OPERATOR.items():
        buat_kartu(nama, info)

    status.pack(pady=(8, 0))

    # ----- Tombol keluar -----
    tombol_keluar = tk.Label(root, text="Keluar  [Esc]", bg="#2a1f2b", fg="#f87171",
                             font=(FONT, 10, "bold"), cursor="hand2", padx=20, pady=8)
    tombol_keluar.pack(pady=(4, 20))
    tombol_keluar.bind("<Enter>", lambda e: tombol_keluar.config(bg="#f87171", fg="white"))
    tombol_keluar.bind("<Leave>", lambda e: tombol_keluar.config(bg="#2a1f2b", fg="#f87171"))
    tombol_keluar.bind("<Button-1>", lambda e: root.destroy())
    root.bind("<Escape>", lambda e: root.destroy())

    # Tengahkan jendela & tampilkan di depan
    root.update_idletasks()
    x = (root.winfo_screenwidth() - root.winfo_width()) // 2
    y = max(10, (root.winfo_screenheight() - root.winfo_height()) // 3)
    root.geometry(f"+{x}+{y}")
    root.attributes("-topmost", True)
    root.after(300, lambda: root.attributes("-topmost", False))

    root.mainloop()
    return pilihan["nama"]


# ---------------------------------------------------------------
# 4. Program utama: menu -> proses -> tampilkan -> kembali ke menu
# ---------------------------------------------------------------
if __name__ == "__main__":
    while True:
        dipilih = menu_pilihan()
        if dipilih is None:
            print("Program selesai.")
            break

        print(f"\n=== Menjalankan {dipilih} ===")
        OPERATOR[dipilih]["fungsi"]()
        plt.show()  # tutup jendela hasil untuk kembali ke menu