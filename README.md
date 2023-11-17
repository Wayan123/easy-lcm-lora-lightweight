# Mudah dan ringan menjalankan Stable Difussion pada CPU

Note: Kode ini sudah dimodifikasi sedemikian rupanya dan disesuaikan dengan kebutuhan.

Repo asli: https://github.com/luosiallen/latent-consistency-model.git 
Huggingface link untuk Super Fast lcm lora stable difussion: https://huggingface.co/spaces/latent-consistency/super-fast-lcm-lora-sd1.5

Pada kode asli dari https://huggingface.co/spaces/latent-consistency/super-fast-lcm-lora-sd1.5 dibuat safety_checker aktif secara default, sedangkan pada kode berikut didisable dengan kode berikut:

def disabled_safety_checker(images, clip_input):
    if len(images.shape) == 4:
        num_images = images.shape[0]
        return images, [False]*num_images
    else:
        return images, False
pipe.safety_checker = disabled_safety_checker

Refrensi kode dari komentar: https://github.com/CompVis/stable-diffusion/issues/331#issuecomment-1562198856

