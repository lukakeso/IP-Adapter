import os
import torch
from diffusers import StableDiffusionInpaintPipelineLegacy, DDIMScheduler, AutoencoderKL, StableDiffusionInpaintPipeline
from PIL import Image
import argparse
import json
from pathlib import Path
from ip_adapter import IPAdapterPlus

def image_grid(imgs, rows, cols):
    assert len(imgs) == rows*cols

    w, h = imgs[0].size
    grid = Image.new('RGB', size=(cols*w, rows*h))
    grid_w, grid_h = grid.size
    
    for i, img in enumerate(imgs):
        grid.paste(img, box=(i%cols*w, i//cols*h))
    return grid

def parse_args():
    parser = argparse.ArgumentParser(description="Inpaining inference script.")
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default="models/SG161222/Realistic_Vision_V4.0_noVAE",
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--pretrained_ip_adapter_root_path",
        type=str,
        default=None,
        required=True,
        help="Path to root folder of ip adapter model.",
    )
    parser.add_argument(
        "--pretrained_vae_path",
        type=str,
        default="models/stabilityai/sd-vae-ft-mse",
        help="Path to pretrained VAE model.",
    )
    parser.add_argument(
        "--num_tokens",
        type=int,
        default=16,
        help="Number of tokens to query from the CLIP image encoding.",
    )
    parser.add_argument(
        "--data_json_file_test",
        type=str,
        default=None,
        required=True,
        help="Testing data",
    )
    parser.add_argument(
        "--data_root_path_test",
        type=str,
        default=None,
        required=True,
        help="Test data root path",
    )
    parser.add_argument(
        "--cloth_mask",
        action='store_true', 
        help="Flag whether to use cloth masks, if not provided agnostic masks will be used",
    )
    
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    # define paths
    
    args = parse_args()
    root = os.getcwd()
    
    direst_dir = "Paired_Direst"
    concat_dir = "Paired_Concat"
    grid_dir = "Paired_Grid"
    
    base_model_path = os.path.join(root, args.pretrained_model_name_or_path)
    vae_model_path = os.path.join(root, args.pretrained_vae_path)
    image_encoder_path = os.path.join(root, "models/image_encoder")
    ip_root = os.path.join(root, args.pretrained_ip_adapter_root_path)
    device = "cuda"
    #seed = 42
    
        
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

    if args.pretrained_model_name_or_path == "models/SG161222/Realistic_Vision_V4.0_noVAE":
        # the original implementation from the paper using Legacy pipeline
        print("using old pipeline")
        pipe = StableDiffusionInpaintPipelineLegacy.from_pretrained(
            base_model_path,
            torch_dtype=torch.float16,
            scheduler=noise_scheduler,
            vae=vae,
            feature_extractor=None,
            safety_checker=None
        )
    else:
        # load up to date SD pipeline with IP-Adapter
        print("using new pipeline")
        pipe = StableDiffusionInpaintPipeline.from_pretrained(
            base_model_path,
            torch_dtype=torch.float16,
            scheduler=noise_scheduler,
            vae=vae,
            feature_extractor=None,
            safety_checker=None
        )        
    
    if isinstance(pipe, StableDiffusionInpaintPipelineLegacy):
        direst_dir = "Paired_Direst_L"
        concat_dir = "Paired_Concat_L"
        grid_dir = "Paired_Grid_L"
    else:
        direst_dir = "Paired_Direst"
        concat_dir = "Paired_Concat"
        grid_dir = "Paired_Grid"

    
    ip_ckpt_list = list() 
    if ip_root.endswith(".bin") and Path(ip_root).exists():
        model_path = ip_root
        save_root = os.path.join("results/ip_adapter_plus/",
                                "authors",                               #training_run
                                args.data_json_file_test.split("/")[1],  #dataset
                                ip_root.split("/")[-1].split(".")[0])    #model
        ip_ckpt_list.append([model_path, save_root])
    else:
        for folder in os.listdir(ip_root):
            if 'checkpoint' in folder:
                
                model_path = os.path.join(ip_root, folder, "ip_adapter_new.bin")
                save_root = os.path.join("results/ip_adapter_plus/",
                                         ip_root.split("/")[-1],                  #training_run
                                         args.data_json_file_test.split("/")[1],  #dataset
                                         folder)                                  #model/checkpoint nr.
                if Path(model_path).exists():
                    ip_ckpt_list.append([model_path, save_root])
  
                
    #generator = torch.Generator(device).manual_seed(seed)
        
    ip_ckpt = os.path.join(root, "models/ip-adapter-plus_sd15_new.bin")
    ip_model = IPAdapterPlus(pipe, image_encoder_path, ip_ckpt, device, num_tokens=args.num_tokens)
    
    data = json.load(open(args.data_json_file_test)) # list of dict: [{"image_file": "1.png", "text": "A dog"}]

    
    for ip_ckpt, save_path in ip_ckpt_list:
        
        os.makedirs(save_path, exist_ok=True)
        os.makedirs(os.path.join(save_path, direst_dir), exist_ok=True)
        os.makedirs(os.path.join(save_path, concat_dir), exist_ok=True)
        os.makedirs(os.path.join(save_path, grid_dir), exist_ok=True)
        
        print(f"Loading: {ip_ckpt}")
        # load ip-adapter
        ip_model.load_ip_adapter(ip_ckpt)
        #ip_model = IPAdapterPlus(pipe, image_encoder_path, ip_ckpt, device, num_tokens=args.num_tokens)
        #pipe.load_ip_adapter(weight_name=ip_ckpt)
        #pipe.set_ip_adapter_scale(0.5)
        
        
        print("Generating Images...")
        for i, item in enumerate(data):
            
            text = item["text"]
            image_file = item["image_file"]
            mask_file = item["mask_file"]
            cloth_file = item["cloth_file"]
            
            if args.cloth_mask:
                mask_file = mask_file.replace("agnostic-mask", "gt_cloth_warped_mask").replace("_mask.png", ".png")
                if not os.path.isfile(os.path.join(args.data_root_path_test, mask_file)):
                    continue
            # read images
            raw_image = Image.open(os.path.join(args.data_root_path_test, image_file)).resize((256, 384))
            mask_image = Image.open(os.path.join(args.data_root_path_test, mask_file)).resize((256, 384))
            ref_image = Image.open(os.path.join(args.data_root_path_test, cloth_file)).resize((256, 256))
            
            orig_images = [raw_image.resize((512, 512)), mask_image.resize((512, 512)), ref_image.resize((512, 512))]

            #save_name = image_file.split("/")[1].split(".")[0]
            
            # only image prompt
            with torch.no_grad():
                # images = pipe(prompt_embeds=prompt_embeds,
                #               negative_prompt_embeds=negative_prompt_embeds,
                #               guidance_scale=guidance_scale,
                #               num_inference_steps=num_inference_steps,
                #               generator=generator,
                #               **kwargs,
                #                 ).images
                images = ip_model.generate(pil_image=ref_image, num_samples=3, num_inference_steps=50,
                                         seed=42, image=raw_image, mask_image=mask_image, strength=0.7)
            images = [img.resize((512, 512)) for img in images]
            
            #save single
            single_img = images[0]
            single_img.save(os.path.join(save_path, direst_dir, str(i)+".png"))
            
            #x_checked_image_torch*(1-mask) + truth.cpu()*mask
            concat_img = Image.composite(single_img, orig_images[0], orig_images[1].convert('L') )
            concat_img.save(os.path.join(save_path, concat_dir, str(i)+".png"))
            
            #save grid
            grid_one = image_grid(orig_images+images, 2, 3)
            grid_one.save(os.path.join(save_path, grid_dir, str(i)+".png"))
            
            # multimodal prompts
            with torch.no_grad():
                # pipe.set_ip_adapter_scale(0.6)
                # images_multi = pipe(prompt_embeds=prompt_embeds,
                #                     negative_prompt_embeds=negative_prompt_embeds,
                #                     guidance_scale=guidance_scale,
                #                     num_inference_steps=num_inference_steps,
                #                     generator=generator,
                #                     **kwargs,
                #                 ).images
                
                images_multi = ip_model.generate(pil_image=ref_image, num_samples=3, num_inference_steps=50,
                                         seed=42, image=raw_image, mask_image=mask_image, strength=0.7,
                                         prompt=text, scale=0.6)
            images = [img.resize((512, 512)) for img in images_multi]
            
            #save single
            single_img = images[0]
            single_img.save(os.path.join(save_path, direst_dir, str(i)+"_m.png"))
            
            #x_checked_image_torch*(1-mask) + truth.cpu()*mask
            concat_img = Image.composite(single_img, orig_images[0], orig_images[1].convert('L'))
            concat_img.save(os.path.join(save_path, concat_dir, str(i)+"_m.png"))
            
            #save grid
            grid_multi = image_grid(orig_images+images, 2, 3)
            grid_multi.save(os.path.join(save_path, grid_dir, str(i)+"_m.png"))        
        
        #print("Moving model to CPU")
        #ip_model.to("cpu")
        print("Deleting model from memory")
        del ip_model
        
            
    print("Done!")
