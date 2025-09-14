# main.py

import utils
import numpy as np
import matplotlib.pyplot as plt

def main():
    # Ganti dengan path gambar Anda (bisa JPG, PNG, dll.)
    image_path = 'Water.png' 

    # 1. Baca gambar dan ubah menjadi grayscale
    print("Membaca dan mengonversi gambar...")
    original_image = utils.imread(image_path)
    if original_image is None:
        return
    gray_image = utils.rgb2grayscale(original_image)

    # 2. Lakukan FFT pada gambar grayscale
    print("Melakukan FFT 2D...")
    frequency_domain_image = utils.fft_2d(gray_image)

    # 3. Siapkan untuk visualisasi FFT
    #    a. Hitung magnitudo dari bilangan kompleks
    #    b. Terapkan skala logaritmik untuk kompresi rentang dinamis
    print("Menghitung spektrum magnitudo untuk visualisasi...")
    magnitude_spectrum = np.log(1 + np.abs(frequency_domain_image))

    # 4. Tampilkan hasilnya berdampingan
    plt.figure(figsize=(12, 6))

    # Gambar Grayscale Asli
    plt.subplot(1, 2, 1)
    plt.imshow(gray_image, cmap='gray')
    plt.title('Gambar Grayscale Asli (Domain Spasial)')
    plt.axis('off')

    # Visualisasi Spektrum Magnitudo
    plt.subplot(1, 2, 2)
    plt.imshow(magnitude_spectrum, cmap='gray')
    plt.title('Spektrum Magnitudo (Domain Frekuensi)')
    plt.axis('off')

    plt.suptitle('Visualisasi Gambar dan Hasil FFT-nya')
    plt.show()

if __name__ == "__main__":
    main()