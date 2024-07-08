import os
import torch
from diffusers import StableDiffusionPipeline, DDIMScheduler, AutoencoderKL
from PIL import Image

from ip_adapter import IPAdapterPlus

def image_grid(imgs, rows, cols):
    assert len(imgs) == rows*cols

    w, h = imgs[0].size
    grid = Image.new('RGB', size=(cols*w, rows*h))
    grid_w, grid_h = grid.size
    
    for i, img in enumerate(imgs):
        grid.paste(img, box=(i%cols*w, i//cols*h))
    return grid

if __name__ == "__main__":
    # define paths
    root = os.getcwd()
    demo_save_path = os.path.join(root, "demos")
    base_model_path = os.path.join(root, "models/SG161222/Realistic_Vision_V4.0_noVAE")
    vae_model_path = os.path.join(root, "models/stabilityai/sd-vae-ft-mse")
    image_encoder_path = os.path.join(root, "models/image_encoder")
    ip_ckpt = os.path.join(root, "models/ip-adapter-plus_sd15.bin")
    device = "cuda"
    
    # define noise scheduler
    noise_scheduler = DDIMScheduler(
        num_train_timesteps=1000,
        beta_start=0.00085,
        beta_end=0.012,
        beta_schedule="scaled_linear",
        clip_sample=False,
        set_alpha_to_one=False,
        steps_offset=1,
    )
    # load VAE 
    vae = AutoencoderKL.from_pretrained(vae_model_path).to(dtype=torch.float16)

    # load SD pipeline
    pipe = StableDiffusionPipeline.from_pretrained(
        base_model_path,
        torch_dtype=torch.float16,
        scheduler=noise_scheduler,
        vae=vae,
        feature_extractor=None,
        safety_checker=None
    )

    # read image prompt
    image = Image.open("assets/images/statue.png")
    image.resize((256, 256))

    # load ip-adapter
    ip_model = IPAdapterPlus(pipe, image_encoder_path, ip_ckpt, device, num_tokens=16)

    print("Generating Images...")
    # only image prompt
    images = ip_model.generate(pil_image=image, num_samples=4, num_inference_steps=50, seed=42)
    grid_one = image_grid(images, 1, 4)
    grid_one.save(os.path.join(demo_save_path, 'img_prompt_grid.png'))

    # multimodal prompts
    images = ip_model.generate(pil_image=image, num_samples=4, num_inference_steps=50, seed=42,
            prompt="best quality, high quality, wearing a hat on the beach", scale=0.6)
    grid_multi = image_grid(images, 1, 4)
    grid_multi.save(os.path.join(demo_save_path, 'multimodal1_grid.png'))

    images = ip_model.generate(pil_image=image, num_samples=4, num_inference_steps=50, seed=42,
            prompt="best quality, high quality, wearing sunglasses in a garden", scale=0.6)
    grid_multi = image_grid(images, 1, 4)
    grid_multi.save(os.path.join(demo_save_path, 'multimodal2_grid.png'))
    print("Done!")
