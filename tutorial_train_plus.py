import os, sys, datetime
import random
import argparse
from pathlib import Path
import json
import itertools
import time
from tqdm import tqdm

import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
from transformers import CLIPImageProcessor
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration
from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel, StableDiffusionInpaintPipelineLegacy
from diffusers import StableDiffusionInpaintPipeline, StableDiffusionControlNetInpaintPipeline
from diffusers import ControlNetModel
from diffusers.pipelines.controlnet.multicontrolnet import MultiControlNetModel

from transformers import CLIPTextModel, CLIPTokenizer, CLIPVisionModelWithProjection, AutoModel

from ip_adapter.resampler import Resampler

from ip_adapter.utils import is_torch2_available
if is_torch2_available():  
    from ip_adapter.attention_processor import IPAttnProcessor2_0 as IPAttnProcessor, AttnProcessor2_0 as AttnProcessor, CNAttnProcessor2_0 as CNAttnProcessor
    from diffusers.models.attention_processor import IPAdapterAttnProcessor2_0 as IPAdapterAttnProcessor#, AttnProcessor2_0 as AttnProcessor
else:
    from ip_adapter.attention_processor import IPAttnProcessor, AttnProcessor, CNAttnProcessor
    from diffusers.models.attention_processor import IPAdapterAttnProcessor#, AttnProcessor

from ip_adapter import IPAdapterPlus as IPAdapterPlus_test

from torchvision.utils import make_grid
from torchvision.transforms.functional import to_pil_image
import wandb

def image_grid(imgs, rows, cols):
    assert len(imgs) == rows*cols

    w, h = imgs[0].size
    grid = Image.new('RGB', size=(cols*w, rows*h))
    grid_w, grid_h = grid.size
    
    for i, img in enumerate(imgs):
        grid.paste(img, box=(i%cols*w, i//cols*h))
    #os.makedirs("results/ip_adapter_plus", exist_ok=True)
    #grid.save(os.path.join("results/ip_adapter_plus", f'grid{idx}.png'))   #careful with idx!
    return grid

def denormalize(tensor):
    return (tensor / 2 + 0.5).clamp(0, 1)


# Dataset
class MyDataset(torch.utils.data.Dataset):

    def __init__(self, json_file, tokenizer, size=512, t_drop_rate=0.05, i_drop_rate=0.05, ti_drop_rate=0.05, image_root_path=""):
        super().__init__()

        self.tokenizer = tokenizer
        self.size = size
        self.i_drop_rate = i_drop_rate
        self.t_drop_rate = t_drop_rate
        self.ti_drop_rate = ti_drop_rate
        self.image_root_path = image_root_path

        self.data = json.load(open(json_file)) # list of dict: [{"image_file": "1.png", "text": "A dog"}]

        self.transform_mask = transforms.Compose([
            transforms.Resize(self.size, interpolation=transforms.InterpolationMode.NEAREST),
            transforms.CenterCrop(self.size),
            transforms.ToTensor(),
        ])
        
        self.transform_im = transforms.Compose([
            transforms.Resize(self.size, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(self.size),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])
        #self.image_processor = CLIPImageProcessor(size={"shortest_edge": 512}, crop_size={"height": 512, "width": 512})
        self.image_processor = CLIPImageProcessor()
        
    def __getitem__(self, idx):
        item = self.data[idx] 
        text = item["text"]
        image_file = item["image_file"]
        mask_file = item["mask_file"]
        cloth_file = item["cloth_file"]
        
        # take agnostic mask not only cloth mask
        mask_file = mask_file.replace("gt_cloth_warped_mask", "agnostic-mask").replace(".jpg", "_mask.png")

        # read images
        raw_image = Image.open(os.path.join(self.image_root_path, image_file))
        mask_image = Image.open(os.path.join(self.image_root_path, mask_file))
        ref_image = Image.open(os.path.join(self.image_root_path, cloth_file))
        densepose_image = Image.open(os.path.join(self.image_root_path, image_file.replace("image", "image-densepose")))
        
        image = self.transform_im(raw_image.convert("RGB"))
        image_mask = self.transform_mask(mask_image.convert("L"))
        clip_image = self.image_processor(images=ref_image, return_tensors="pt").pixel_values
        densepose = self.transform_mask(densepose_image)
        control_image = torch.cat((transforms.Resize((512, 512))(clip_image[0]),densepose),dim = 0)
        
        # drop
        drop_image_embed = 0
        rand_num = random.random()
        if rand_num < self.i_drop_rate:
            drop_image_embed = 1
        elif rand_num < (self.i_drop_rate + self.t_drop_rate):
            text = ""
        elif rand_num < (self.i_drop_rate + self.t_drop_rate + self.ti_drop_rate):
            text = ""
            drop_image_embed = 1
        # get text and tokenize
        text_input_ids = self.tokenizer(
            text,
            max_length=self.tokenizer.model_max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        ).input_ids
        
        return {
            "image": image,
            "text_input_ids": text_input_ids,
            "clip_image": clip_image,
            "drop_image_embed": drop_image_embed,
            "image_mask": image_mask,
            "img_desc": text,
            "image_control": control_image,
        }

    def __len__(self):
        return len(self.data)
    

def collate_fn(data):
    images = torch.stack([example["image"] for example in data])
    images_mask = torch.stack([example["image_mask"] for example in data])
    text_input_ids = torch.cat([example["text_input_ids"] for example in data], dim=0)
    clip_images = torch.cat([example["clip_image"] for example in data], dim=0)
    drop_image_embeds = [example["drop_image_embed"] for example in data]
    texts = [example["img_desc"] for example in data]
    control_images = torch.stack([example["image_control"] for example in data])
    
    return {
        "images": images,
        "text_input_ids": text_input_ids,
        "clip_images": clip_images,
        "drop_image_embeds": drop_image_embeds,
        "images_mask": images_mask,
        "img_descs": texts,
        "images_control": control_images,
    }

def preprocess_mask(mask, scale_factor=8):
    if not isinstance(mask, torch.Tensor):
        mask = mask.convert("L")
        w, h = mask.size
        w, h = (x - x % 8 for x in (w, h))  # resize to integer multiple of 8
        mask = mask.resize((w // scale_factor, h // scale_factor), resample=Image.NEAREST)
        mask = np.array(mask).astype(np.float32) / 255.0
        mask = np.tile(mask, (4, 1, 1))
        #mask = np.vstack([mask[None]] * batch_size)
        mask = 1 - mask  # repaint white, keep black
        mask = torch.from_numpy(mask)
        return mask

    else:
        valid_mask_channel_sizes = [1, 3]
        # if mask channel is fourth tensor dimension, permute dimensions to pytorch standard (B, C, H, W)
        if mask.shape[3] in valid_mask_channel_sizes:
            mask = mask.permute(0, 3, 1, 2)
        elif mask.shape[1] not in valid_mask_channel_sizes:
            raise ValueError(
                f"Mask channel dimension of size in {valid_mask_channel_sizes} should be second or fourth dimension,"
                f" but received mask of shape {tuple(mask.shape)}"
            )
        # (potentially) reduce mask channel dimension from 3 to 1 for broadcasting to latent shape
        mask = mask.mean(dim=1, keepdim=True)
        h, w = mask.shape[-2:]
        h, w = (x - x % 8 for x in (h, w))  # resize to integer multiple of 8
        mask = torch.nn.functional.interpolate(mask, (h // scale_factor, w // scale_factor))
        return mask

class IPAdapter(torch.nn.Module):
    """IP-Adapter"""
    def __init__(self, unet, image_proj_model, adapter_modules, ckpt_path=None, controlnet=None):
        super().__init__()
        self.unet = unet
        self.controlnet = controlnet if controlnet != None else None
        self.image_proj_model = image_proj_model
        self.adapter_modules = adapter_modules

        if ckpt_path is not None:
            self.load_from_checkpoint(ckpt_path)

    
    def forward(self, noisy_latents, mask, masked_images_latents, timesteps, prompt_embeds, image_embeds, control_images=None):
        ip_embeds = self.image_proj_model(image_embeds)
        
        # concatenate latents and masks for 9 channel unet/controlnet
        latent_model_input = torch.cat([noisy_latents, mask, masked_images_latents], dim=1)
        
        if self.controlnet == None:
            down_block_res_samples, mid_block_res_sample = None, None
        else:
            down_block_res_samples, mid_block_res_sample = self.controlnet(
                latent_model_input,
                timesteps,
                encoder_hidden_states=prompt_embeds,
                controlnet_cond=control_images,
                conditioning_scale=0.5, # what should this number be?
                guess_mode=False,
                return_dict=False,
            )
        
        # Predict the noise residual
        noise_pred = self.unet(latent_model_input, timesteps, 
                               encoder_hidden_states=prompt_embeds,
                               down_block_additional_residuals=down_block_res_samples,
                               mid_block_additional_residual=mid_block_res_sample,
                               added_cond_kwargs={"image_embeds": ip_embeds}).sample
        # inputs to unet from controlnet
        #down_block_additional_residuals=down_block_res_samples,
        #mid_block_additional_residual=mid_block_res_sample,
                
        return noise_pred

    def load_from_checkpoint(self, ckpt_path: str):
        # Calculate original checksums
        orig_ip_proj_sum = torch.sum(torch.stack([torch.sum(p) for p in self.image_proj_model.parameters()]))
        orig_adapter_sum = torch.sum(torch.stack([torch.sum(p) for p in self.adapter_modules.parameters()]))

        state_dict = torch.load(ckpt_path, map_location="cpu")

        # Check if 'latents' exists in both the saved state_dict and the current model's state_dict
        strict_load_image_proj_model = True
        if "latents" in state_dict["image_proj"] and "latents" in self.image_proj_model.state_dict():
            # Check if the shapes are mismatched
            if state_dict["image_proj"]["latents"].shape != self.image_proj_model.state_dict()["latents"].shape:
                print(f"Shapes of 'image_proj.latents' in checkpoint {ckpt_path} and current model do not match.")
                print("Removing 'latents' from checkpoint and loading the rest of the weights.")
                del state_dict["image_proj"]["latents"]
                strict_load_image_proj_model = False

        # Load state dict for image_proj_model and adapter_modules
        self.image_proj_model.load_state_dict(state_dict["image_proj"], strict=strict_load_image_proj_model)
        self.adapter_modules.load_state_dict(state_dict["ip_adapter"], strict=True)

        # Calculate new checksums
        new_ip_proj_sum = torch.sum(torch.stack([torch.sum(p) for p in self.image_proj_model.parameters()]))
        new_adapter_sum = torch.sum(torch.stack([torch.sum(p) for p in self.adapter_modules.parameters()]))

        # Verify if the weights have changed
        assert orig_ip_proj_sum != new_ip_proj_sum, "Weights of image_proj_model did not change!"
        assert orig_adapter_sum != new_adapter_sum, "Weights of adapter_modules did not change!"

        print(f"Successfully loaded weights from checkpoint {ckpt_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default=None,
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--pretrained_ip_adapter_path",
        type=str,
        default=None,
        help="Path to pretrained ip adapter model. If not specified weights are initialized randomly.",
    )
    parser.add_argument(
        "--pretrained_controlnet_path",
        type=str,
        default=None,
        help="Path to pretrained controlnet model. If not specified weights are not initialized.",
    )
    parser.add_argument(
        "--num_tokens",
        type=int,
        default=16,
        help="Number of tokens to query from the CLIP image encoding.",
    )
    parser.add_argument(
        "--data_json_file_train",
        type=str,
        default=None,
        required=True,
        help="Training data",
    )
    parser.add_argument(
        "--data_json_file_test",
        type=str,
        default=None,
        required=False,
        help="Testing data",
    )
    parser.add_argument(
        "--data_root_path_train",
        type=str,
        default="",
        required=True,
        help="Training data root path",
    )
    parser.add_argument(
        "--data_root_path_test",
        type=str,
        default="",
        required=False,
        help="Test data root path",
    )
    parser.add_argument(
        "--image_encoder_path",
        type=str,
        default=None,
        required=True,
        help="Path to CLIP image encoder",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="sd-ip_adapter",
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
        help=(
            "[TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to"
            " *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***."
        ),
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=512,
        help=(
            "The resolution for input images"
        ),
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
        help="Learning rate to use.",
    )
    parser.add_argument("--weight_decay", type=float, default=1e-2, help="Weight decay to use.")
    parser.add_argument("--num_train_epochs", type=int, default=100)
    parser.add_argument(
        "--train_batch_size", type=int, default=8, help="Batch size (per device) for the training dataloader."
    )
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=1,
        help=(
            "Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process."
        ),
    )
    parser.add_argument(
        "--save_steps",
        type=int,
        default=2000,
        help=(
            "Save a checkpoint of the training state every X updates"
        ),
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default=None,
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
            " 1.10.and an Nvidia Ampere GPU.  Default to the value of accelerate config of the current system or the"
            " flag passed with the `accelerate.launch` command. Use this argument to override the accelerate config."
        ),
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="all",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`'
            ' (default), `"wandb"` and `"comet_ml"`. Use `"all"` to report to all integrations.'
        ),
    )
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")
    parser.add_argument("--trial_run", action='store_true', help="A trial run, no training happens")
    parser.add_argument("--wandb_project", type=str, default="IP-Adapter", help="Name for the WANDB project")

    args = parser.parse_args()
    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    return args

def main():
    print("staring main...")
    args = parse_args()
    dir_i = 0
    now = datetime.datetime.now().strftime("%Y-%m-%dT%H")
    
    while Path(args.output_dir+"_"+now+"_"+str(dir_i)).exists():
        dir_i += 1
    
    args.output_dir = args.output_dir+"_"+now+"_"+str(dir_i)
    logging_dir = Path(args.output_dir, args.logging_dir)
    #wandb_logging_dir = Path(os.getcwd(), args.output_dir)
    
    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)

    accelerator = Accelerator(
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
    )
    accelerator.init_trackers(
        project_name=args.wandb_project, 
        config=vars(args),
        init_kwargs={"wandb": {"entity": "lukak",
                               "tags": [str(args.learning_rate),                    #learning rate
                                        args.data_json_file_train.split("/")[1],    #dataset
                                        "agnostic_masking",                         #type of masking
                                        "control_sd"],                              #model (sd/pbe)
                               "name": now+"_"+str(dir_i),
                               "dir": logging_dir}}
    )
    
    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)

    # accelerator.wait_for_everyone()
    # Load scheduler, tokenizer and models.
    noise_scheduler = DDPMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")
    tokenizer = CLIPTokenizer.from_pretrained(args.pretrained_model_name_or_path, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="text_encoder")
    vae = AutoencoderKL.from_pretrained("models/stabilityai/sd-vae-ft-mse", subfolder="vae")
    #vae = AutoencoderKL.from_pretrained(args.pretrained_model_name_or_path, subfolder="vae")
    unet = UNet2DConditionModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="unet")
    #image_encoder = CLIPVisionModelWithProjection.from_pretrained(args.image_encoder_path)
    image_encoder = AutoModel.from_pretrained(args.image_encoder_path)
    
    #controlnet = ControlNetModel.from_single_file("/d/hpc/home/lk6760/FRI_HOME/PRETRAINED_MODELS/dense_control.safetensors") #4channel
    controlnet = None
    if args.pretrained_controlnet_path:
        controlnet = ControlNetModel.from_single_file(args.pretrained_controlnet_path, 
                                                      config_file=args.pretrained_controlnet_path.replace(".ckpt", ".yaml"))
    
    # pretrained_model_name_or_path = "models/runwayml/stable-diffusion-inpainting"
    # pipe = StableDiffusionInpaintPipeline.from_pretrained(
    #     args.pretrained_model_name_or_path,
    #     scheduler=noise_scheduler,
    #     vae=vae,
    #     unet=unet,
    #     feature_extractor=None,
    #     safety_checker=None,
    #     torch_dtype=torch.float16
    # )
    if controlnet == None:
        pipe = StableDiffusionInpaintPipeline(
            scheduler=noise_scheduler,
            vae=vae,
            unet=unet,
            text_encoder=None,
            tokenizer=None,
            feature_extractor=None,
            safety_checker=None
            #torch_dtype=torch.float16
        )
    else:
        pipe = StableDiffusionControlNetInpaintPipeline(
            scheduler=noise_scheduler,
            vae=vae,
            unet=unet,
            controlnet=controlnet,
            text_encoder=None,
            tokenizer=None,
            feature_extractor=None,
            safety_checker=None,
            #torch_dtype=torch.float16
        )
    
    # freeze parameters of models to save more memory
    unet.requires_grad_(False)
    pipe.unet.requires_grad_(False)
    pipe.vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    image_encoder.requires_grad_(False)
    
    #ip-adapter-plus
    image_proj_model = Resampler(
        dim=unet.config.cross_attention_dim,
        depth=4,
        dim_head=64,
        heads=12,
        num_queries=args.num_tokens,
        embedding_dim=image_encoder.config.hidden_size,
        output_dim=unet.config.cross_attention_dim,
        ff_mult=4
    )
    
    # load SD pipeline
    # init adapter modules    
    attn_procs = {}
    unet_sd = unet.state_dict()
    for name in unet.attn_processors.keys():
        cross_attention_dim = None if name.endswith("attn1.processor") else unet.config.cross_attention_dim
        if name.startswith("mid_block"):
            hidden_size = unet.config.block_out_channels[-1]
        elif name.startswith("up_blocks"):
            block_id = int(name[len("up_blocks.")])
            hidden_size = list(reversed(unet.config.block_out_channels))[block_id]
        elif name.startswith("down_blocks"):
            block_id = int(name[len("down_blocks.")])
            hidden_size = unet.config.block_out_channels[block_id]
        if cross_attention_dim is None:
            attn_procs[name] = AttnProcessor()
        else:
            layer_name = name.split(".processor")[0]
            weights = {
                "to_k_ip.0.weight": unet_sd[layer_name + ".to_k.weight"],
                "to_v_ip.0.weight": unet_sd[layer_name + ".to_v.weight"],
            }
            #attn_procs[name] = IPAttnProcessor(hidden_size=hidden_size, cross_attention_dim=cross_attention_dim, num_tokens=args.num_tokens)
            attn_procs[name] = IPAdapterAttnProcessor(hidden_size=hidden_size, cross_attention_dim=cross_attention_dim, num_tokens=args.num_tokens)
            #attn_procs[name] = IPAdapterAttnProcessor(hidden_size=hidden_size, cross_attention_dim=cross_attention_dim, num_tokens=16)
            attn_procs[name].load_state_dict(weights)
    if hasattr(pipe, "controlnet"):
        if isinstance(pipe.controlnet, MultiControlNetModel):
            for controlnet in pipe.controlnet.nets:
                controlnet.set_attn_processor(CNAttnProcessor(num_tokens=args.num_tokens))
        else:
            pipe.controlnet.set_attn_processor(CNAttnProcessor(num_tokens=args.num_tokens))
    
    unet.set_attn_processor(attn_procs)
    adapter_modules = torch.nn.ModuleList(unet.attn_processors.values())
    
    ip_adapter = IPAdapter(unet, image_proj_model, adapter_modules, 
                           ckpt_path=args.pretrained_ip_adapter_path, 
                           controlnet=controlnet)

    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16
    
    pipe.to(accelerator.device, dtype=weight_dtype)
    #ip_adapter.to(accelerator.device, dtype=weight_dtype)
    #unet.to(accelerator.device, dtype=weight_dtype)
    vae.to(accelerator.device, dtype=weight_dtype)
    text_encoder.to(accelerator.device, dtype=weight_dtype)
    image_encoder.to(accelerator.device, dtype=weight_dtype)
    
    # optimizer 
    params_to_opt = itertools.chain(ip_adapter.image_proj_model.parameters(),  ip_adapter.adapter_modules.parameters())
    optimizer = torch.optim.AdamW(params_to_opt, lr=args.learning_rate, weight_decay=args.weight_decay)
    
    # dataloader
    train_dataset = MyDataset(args.data_json_file_train, tokenizer=tokenizer, size=args.resolution, image_root_path=args.data_root_path_train)
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        shuffle=True,
        collate_fn=collate_fn,
        batch_size=args.train_batch_size,
        num_workers=args.dataloader_num_workers,
    )
    ip_model_test = IPAdapterPlus_test(pipe, image_encoder, 
                                        args.pretrained_ip_adapter_path, 
                                        device="cuda",
                                        dtype=weight_dtype,
                                        num_tokens=16)
    # Prepare everything with our `accelerator`.
    ip_adapter, optimizer, train_dataloader = accelerator.prepare(ip_adapter, optimizer, train_dataloader)
    
    global_step = 0
    vae_scaling_factor = 2 ** (len(vae.config.block_out_channels) - 1)
    height =  unet.config.sample_size * vae_scaling_factor
    width = unet.config.sample_size * vae_scaling_factor

    if accelerator.is_main_process:
        negative_text_input_ids = tokenizer(
                                "monochrome, lowres, bad anatomy, worst quality, low quality",
                                max_length=tokenizer.model_max_length,
                                padding="max_length",
                                truncation=True,
                                return_tensors="pt"
                            ).input_ids
        negative_prompt_embeds=text_encoder(negative_text_input_ids.to(accelerator.device))[0]
        
        test_dataset = MyDataset(args.data_json_file_test, tokenizer=tokenizer, size=512, image_root_path=args.data_root_path_test)
        test_dataloader = torch.utils.data.DataLoader(
            test_dataset,
            shuffle=False,
            collate_fn=collate_fn,
            batch_size=1,
            num_workers=1
        )
        
        print("logging samples...")
        grids = []

        for test_sample in test_dataloader:
            #image_embeds = image_encoder(test_sample["clip_images"].to(accelerator.device, dtype=weight_dtype), output_hidden_states=True).hidden_states[-2]
            with torch.no_grad():
                orig_images = [to_pil_image(denormalize(test_sample["images"].squeeze()), mode="RGB").resize((512, 512)), 
                           to_pil_image(test_sample["images_mask"].squeeze(), mode="L").resize((512, 512)), 
                           to_pil_image(denormalize(test_sample["clip_images"].squeeze()), mode="RGB").resize((512, 512))]
                
                
                clip_image_embeds = image_encoder(test_sample["clip_images"].to(accelerator.device, dtype=weight_dtype), output_hidden_states=True).hidden_states[-2]
                uncond_clip_image_embeds = image_encoder(torch.zeros_like(test_sample["clip_images"]).to(accelerator.device, dtype=weight_dtype), output_hidden_states=True).hidden_states[-2]
                clip_image_embeds = (clip_image_embeds, uncond_clip_image_embeds)
                
                generated_images = ip_model_test.generate(clip_image_embeds=clip_image_embeds, num_samples=3, num_inference_steps=50,
                        seed=42, image=denormalize(test_sample["images"]), mask_image=test_sample["images_mask"], strength=0.7, 
                        prompt_embeds=text_encoder(test_sample["text_input_ids"].to(accelerator.device))[0],
                        negative_prompt_embeds=negative_prompt_embeds, scale=0.6,
                        control_image=test_sample["images_control"])
                
                generated_images = [img.resize((512,512)) for img in generated_images]
            
            #grid = make_grid(orig_images+generated_images, nrow=3)
            grid = image_grid(orig_images+generated_images, 2, 3)
            grids.append(wandb.Image(grid, caption="Top: Inputs, Bottom: Outputs"))
        accelerator.log({"start_samples": grids}, step=global_step)
            
        print("starting training...")  
        progress_bar = tqdm(total=len(train_dataloader), desc="Training: ", file=sys.stdout)

    accelerator.wait_for_everyone()
    
    for epoch in range(0, args.num_train_epochs):
        #begin = time.perf_counter()
        # ip_adapter.train()
        if accelerator.is_main_process:
            time.sleep(0.3)
            print("Epoch: ", epoch)
            progress_bar.reset()
        accelerator.wait_for_everyone()
        for step, batch in enumerate(train_dataloader):
            #load_data_time = time.perf_counter() - begin
            
            with accelerator.accumulate(ip_adapter):
                # Convert images to latent space
                with torch.no_grad():
                    images_latents = vae.encode(batch["images"].to(accelerator.device, dtype=weight_dtype)).latent_dist.sample()
                    images_latents = images_latents * vae.config.scaling_factor
                    
                    #mask_image = preprocess_mask(batch["images_mask"], vae_scaling_factor)
                    masked_images = batch["images"] * (batch["images_mask"] < 0.5)
                    masked_images_latents = vae.encode(masked_images.to(accelerator.device, dtype=weight_dtype)).latent_dist.sample()
                    masked_images_latents = masked_images_latents * vae.config.scaling_factor
                                        
                    mask = torch.nn.functional.interpolate(
                        batch["images_mask"], size=(height // vae_scaling_factor, width // vae_scaling_factor)
                    )
                    mask.to(device=accelerator.device, dtype=weight_dtype)
                # Sample noise that we'll add to the latents
                noise = torch.randn_like(images_latents)
                bsz = images_latents.shape[0]
                # Sample a random timestep for each image
                timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=accelerator.device)
                timesteps = timesteps.long()

                # Add noise to the latents according to the noise magnitude at each timestep
                # (this is the forward diffusion process)
                noisy_latents = noise_scheduler.add_noise(images_latents, noise, timesteps)
                
                clip_images = []
                for clip_image, drop_image_embed in zip(batch["clip_images"], batch["drop_image_embeds"]):
                    if drop_image_embed == 1:
                        clip_images.append(torch.zeros_like(clip_image))
                    else:
                        clip_images.append(clip_image)
                clip_images = torch.stack(clip_images, dim=0)
                with torch.no_grad():
                    image_embeds = image_encoder(clip_images.to(accelerator.device, dtype=weight_dtype), output_hidden_states=True).hidden_states[-2]
                    control_images = batch["images_control"].to(accelerator.device, dtype=weight_dtype)
                with torch.no_grad():
                    prompt_embeds = text_encoder(batch["text_input_ids"].to(accelerator.device))[0]
                
                # denoising step
                noise_pred = ip_adapter(noisy_latents, mask, masked_images_latents, 
                                        timesteps, prompt_embeds, image_embeds,
                                        control_images=control_images)
                
                # loss
                #loss = F.mse_loss(masked_noise_pred.float(), noise.float(), reduction="mean")
                loss = F.mse_loss(noise_pred.float(), noise.float(), reduction="mean")
            
                # Gather the losses across all processes for logging (if we use distributed training).
                avg_loss = accelerator.gather(loss.repeat(args.train_batch_size)).mean().item()
                
                # Backpropagate
                accelerator.backward(loss)
                optimizer.step()
                optimizer.zero_grad()

                if accelerator.is_main_process:
                    progress_bar.update(1)
                    accelerator.log({"epoch": epoch, "train_loss": avg_loss}, step=global_step)
                    # print("Epoch {}, step {}, data_time: {}, time: {}, step_loss: {}".format(
                    #      epoch, step, load_data_time, time.perf_counter() - begin, avg_loss))
            
            global_step += 1
            
            if global_step % args.save_steps == 0:
                save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                accelerator.wait_for_everyone()
                accelerator.save_state(save_path)
                accelerator.wait_for_everyone()
                ip_model_test.load_ip_adapter(os.path.join(save_path, "model.safetensors"))

                if accelerator.is_main_process:
                    print("logging samples...")
                    grids = []
                    for test_sample in test_dataloader:
                        
                        #image_embeds = image_encoder(test_sample["clip_images"].to(accelerator.device, dtype=weight_dtype), output_hidden_states=True).hidden_states[-2]
                        with torch.no_grad():
                            # orig_images = [to_pil_image(denormalize(test_sample["images"].squeeze()), mode="RGB").resize((512, 512)), 
                            #                to_pil_image(test_sample["images_mask"].squeeze(), mode="L").resize((512, 512)), 
                            #                to_pil_image(denormalize(test_sample["clip_images"].squeeze()), mode="RGB").resize((512, 512))]
                            clip_image_embeds = image_encoder(test_sample["clip_images"].to(accelerator.device, dtype=weight_dtype), output_hidden_states=True).hidden_states[-2]
                            uncond_clip_image_embeds = image_encoder(torch.zeros_like(test_sample["clip_images"]).to(accelerator.device, dtype=weight_dtype), output_hidden_states=True).hidden_states[-2]
                            clip_image_embeds = (clip_image_embeds, uncond_clip_image_embeds)
                            
                            generated_images = ip_model_test.generate(clip_image_embeds=clip_image_embeds, num_samples=3, num_inference_steps=50,
                                    seed=42, image=denormalize(test_sample["images"]), mask_image=test_sample["images_mask"], strength=0.7, 
                                    prompt_embeds=text_encoder(test_sample["text_input_ids"].to(accelerator.device))[0],
                                    negative_prompt_embeds=negative_prompt_embeds, scale=0.6,
                                    control_image=test_sample["images_control"])
                
                            generated_images = [img.resize((512, 512)) for img in generated_images]
            
                        #grid = make_grid(orig_images+generated_images, nrow=3)   
                        grid = image_grid(generated_images, 1, 3)
                        grids.append(wandb.Image(grid, caption="generated outputs"))
                    
                    accelerator.log({"test_samples": grids}, step=global_step)
                accelerator.wait_for_everyone()
                
            
            #begin = time.perf_counter()
    if accelerator.is_main_process:
        print(f'Finished training after {epoch+1} epochs ({global_step} steps)') 
    save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}-final")
    accelerator.wait_for_everyone()
    accelerator.save_state(save_path)
    accelerator.wait_for_everyone()    
    accelerator.end_training()

if __name__ == "__main__":
    main()
    print("Exiting...")
