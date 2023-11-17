from diffusers import DiffusionPipeline, LCMScheduler
from PIL import Image
import time
from googletrans import Translator  # Menggunakan library Google Translate

translator = Translator()

pipe = DiffusionPipeline.from_pretrained("Lykon/dreamshaper-7")
pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)
pipe.load_lora_weights("latent-consistency/lcm-lora-sdv1-5")

def disabled_safety_checker(images, clip_input):
    if len(images.shape) == 4:
        num_images = images.shape[0]
        return images, [False]*num_images
    else:
        return images, False
pipe.safety_checker = disabled_safety_checker

while True:
    print("")
    # Menerima prompt dalam bahasa Indonesia
    user_prompt_id = input("Masukkan prompt untuk inferensi (atau ketik 'exit' untuk keluar): ")
    
    # Keluar dari loop jika pengguna memasukkan 'exit'
    if user_prompt_id.lower() == 'exit':
        break

    # Terjemahkan prompt ke bahasa Inggris
    user_prompt_en = translator.translate(user_prompt_id, src='id', dest='en').text
    print("Terjemahan: ", user_prompt_en)

    num_inference_steps = int(input("Masukkan jumlah langkah inferensi (num_inference_steps): "))
    # num_photos = int(input("Masukkan jumlah foto yang diinginkan: "))

    start_time = time.time()

    # Menjalankan inferensi dengan prompt yang sudah diterjemahkan
    results = pipe(
        prompt=user_prompt_en,
        num_inference_steps=num_inference_steps,
        guidance_scale=0.3,
        nsfw=False
    )

    end_time = time.time()

    latency_seconds = end_time - start_time
    latency_minutes = int(latency_seconds // 60)
    remaining_seconds = latency_seconds % 60

    print(f"Latensi: {latency_minutes} menit {remaining_seconds:.2f} detik")

    # for i in range(min(num_photos, len(results.images))):
        # results.images[i].show()
        
    results.images[0].show()

