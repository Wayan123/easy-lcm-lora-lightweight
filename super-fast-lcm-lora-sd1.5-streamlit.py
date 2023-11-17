import streamlit as st
from diffusers import DiffusionPipeline, LCMScheduler
from PIL import Image
import time
import io
import os

# Inisialisasi pipeline
pipe = DiffusionPipeline.from_pretrained("Lykon/dreamshaper-7")
pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)
pipe.load_lora_weights("latent-consistency/lcm-lora-sdv1-5")

# Fungsi untuk melakukan inferensi dan menampilkan hasil
def perform_inference(prompt):
    start_time = time.time()

    if 'lcm_diffusion_setting' in pipe.config:
        pipe.config['lcm_diffusion_setting']['use_safety_checker'] = False

    # Menggunakan st.progress() untuk menampilkan progress bar
    progress_bar = st.progress(0.0)

    # Menentukan jumlah langkah inferensi
    num_inference_steps = 4

    for step in range(num_inference_steps):
        # Mengupdate progress bar pada setiap langkah
        progress_bar.progress((step + 1) / num_inference_steps)
        # Menampilkan persentase di progress bar
        progress_text = f"{((step + 1) / num_inference_steps) * 100:.2f}%"
        st.text(progress_text)

        # Melakukan langkah inferensi
        results = pipe(
            prompt=prompt,
            num_inference_steps=4,  # Setiap langkah diambil satu per satu
            guidance_scale=0.3,
            nsfw=False
        )

    end_time = time.time()
    latency_seconds = end_time - start_time

    latency_minutes = int(latency_seconds // 60)
    remaining_seconds = latency_seconds % 60

    st.write(f"Latensi: {latency_minutes} menit {remaining_seconds:.2f} detik")

    # Cek apakah NSFW content terdeteksi
    if hasattr(results, 'warnings') and "Potential NSFW content" in results.warnings:
        st.warning("Potensi konten NSFW terdeteksi. Cobalah lagi dengan prompt dan/atau seed yang berbeda.")
    else:
        # Simpan gambar ke file sementara
        temp_image_path = "temp_image.png"
        results.images[0].save(temp_image_path)

        # Menampilkan gambar dari file sementara
        st.image(temp_image_path, caption='Hasil Inference', use_column_width=True)

        # Hapus file sementara setelah ditampilkan
        os.remove(temp_image_path)

# Streamlit UI
st.title('Streamlit Inference App')

# Meminta pengguna untuk memasukkan prompt
user_prompt = st.text_input("Masukkan prompt untuk inferensi (atau ketik 'exit' untuk keluar): ")

# Check jika pengguna memasukkan 'exit'
if user_prompt.lower() != 'exit':
    # Tombol untuk menjalankan inferensi
    if st.button('Jalankan Inferensi'):
        # Memanggil fungsi perform_inference
        perform_inference(user_prompt)
