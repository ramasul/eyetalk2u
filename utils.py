import numpy as np
from PIL import Image

"""
Modul ini berisi fungsi-fungsi pembantu untuk pemrosesan gambar dasar,
termasuk membaca gambar, konversi ke grayscale, dan transformasi Fourier.
"""

def imread(image_path: str) -> np.ndarray:
    """Membaca file gambar dari path dan mengonversinya menjadi array NumPy."""
    try:
        img = Image.open(image_path)
        return np.array(img)
    except FileNotFoundError:
        print(f"Error: File tidak ditemukan di '{image_path}'")
        return None

def rgb2grayscale(image: np.ndarray) -> np.ndarray:
    """
    Mengonversi gambar RGB atau RGBA menjadi grayscale.
    Fungsi ini aman untuk gambar yang sudah grayscale.
    """
    # Kasus 1: Gambar sudah dalam format grayscale (2 dimensi)
    if image.ndim == 2:
        print("Info: Gambar sudah dalam format grayscale.")
        return image

    # Kasus 2: Gambar adalah 3 dimensi (kemungkinan berwarna)
    elif image.ndim == 3:
        # Jika gambar memiliki 4 channel (RGBA), abaikan channel alpha (transparansi)
        if image.shape[2] == 4:
            print("Info: Gambar adalah RGBA. Channel alpha akan diabaikan.")
            image = image[..., :3]  # Ambil 3 channel pertama (RGB)
        
        # Setelah memastikan gambar adalah RGB (3 channel), lakukan konversi
        if image.shape[2] == 3:
            rgb_weights = np.array([0.299, 0.587, 0.114])
            return np.dot(image, rgb_weights)
        else:
            # Jika channel bukan 3 atau 4, format tidak didukung
            raise ValueError(f"Input gambar 3D memiliki {image.shape[2]} channel, format tidak didukung.")

    # Kasus 3: Dimensi gambar tidak didukung
    else:
        raise ValueError(f"Dimensi gambar tidak didukung: {image.ndim}. Hanya gambar 2D dan 3D yang diterima.")


def fft_2d(image: np.ndarray) -> np.ndarray:
    """Melakukan FFT 2D pada gambar dan memindahkan frekuensi nol ke tengah."""
    f_transform = np.fft.fft2(image)
    return np.fft.fftshift(f_transform)

def ifft_2d(f_transform_shifted: np.ndarray) -> np.ndarray:
    """Melakukan IFFT 2D dan mengembalikan bagian real dari gambar."""
    f_transform = np.fft.ifftshift(f_transform_shifted)
    image_reconstructed = np.fft.ifft2(f_transform)
    return np.real(image_reconstructed)