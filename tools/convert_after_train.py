import os
import torch
from safetensors.torch import load_file
import argparse
from pathlib import Path

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="Model converstion script (from safetensors to bin)")

    parser.add_argument(
        "--pretrained_ip_adapter_root_path",
        type=str,
        default=None,
        required=True,
        help="Path to root folder of ip adapter model, with different checkpoint folders",
    )
    
    args = parser.parse_args()
    
    names_1 = ['down_blocks.0.attentions.0.transformer_blocks.0.attn2.processor.to_k_ip.weight',
               'down_blocks.0.attentions.0.transformer_blocks.0.attn2.processor.to_v_ip.weight',
               'down_blocks.0.attentions.1.transformer_blocks.0.attn2.processor.to_k_ip.weight', 
               'down_blocks.0.attentions.1.transformer_blocks.0.attn2.processor.to_v_ip.weight', 
               'down_blocks.1.attentions.0.transformer_blocks.0.attn2.processor.to_k_ip.weight', 
               'down_blocks.1.attentions.0.transformer_blocks.0.attn2.processor.to_v_ip.weight', 
               'down_blocks.1.attentions.1.transformer_blocks.0.attn2.processor.to_k_ip.weight', 
               'down_blocks.1.attentions.1.transformer_blocks.0.attn2.processor.to_v_ip.weight', 
               'down_blocks.2.attentions.0.transformer_blocks.0.attn2.processor.to_k_ip.weight', 
               'down_blocks.2.attentions.0.transformer_blocks.0.attn2.processor.to_v_ip.weight', 
               'down_blocks.2.attentions.1.transformer_blocks.0.attn2.processor.to_k_ip.weight', 
               'down_blocks.2.attentions.1.transformer_blocks.0.attn2.processor.to_v_ip.weight', 
               'up_blocks.1.attentions.0.transformer_blocks.0.attn2.processor.to_k_ip.weight', 
               'up_blocks.1.attentions.0.transformer_blocks.0.attn2.processor.to_v_ip.weight', 
               'up_blocks.1.attentions.1.transformer_blocks.0.attn2.processor.to_k_ip.weight', 
               'up_blocks.1.attentions.1.transformer_blocks.0.attn2.processor.to_v_ip.weight', 
               'up_blocks.1.attentions.2.transformer_blocks.0.attn2.processor.to_k_ip.weight', 
               'up_blocks.1.attentions.2.transformer_blocks.0.attn2.processor.to_v_ip.weight', 
               'up_blocks.2.attentions.0.transformer_blocks.0.attn2.processor.to_k_ip.weight', 
               'up_blocks.2.attentions.0.transformer_blocks.0.attn2.processor.to_v_ip.weight', 
               'up_blocks.2.attentions.1.transformer_blocks.0.attn2.processor.to_k_ip.weight', 
               'up_blocks.2.attentions.1.transformer_blocks.0.attn2.processor.to_v_ip.weight', 
               'up_blocks.2.attentions.2.transformer_blocks.0.attn2.processor.to_k_ip.weight', 
               'up_blocks.2.attentions.2.transformer_blocks.0.attn2.processor.to_v_ip.weight', 
               'up_blocks.3.attentions.0.transformer_blocks.0.attn2.processor.to_k_ip.weight', 
               'up_blocks.3.attentions.0.transformer_blocks.0.attn2.processor.to_v_ip.weight', 
               'up_blocks.3.attentions.1.transformer_blocks.0.attn2.processor.to_k_ip.weight', 
               'up_blocks.3.attentions.1.transformer_blocks.0.attn2.processor.to_v_ip.weight', 
               'up_blocks.3.attentions.2.transformer_blocks.0.attn2.processor.to_k_ip.weight', 
               'up_blocks.3.attentions.2.transformer_blocks.0.attn2.processor.to_v_ip.weight', 
               'mid_block.attentions.0.transformer_blocks.0.attn2.processor.to_k_ip.weight', 
               'mid_block.attentions.0.transformer_blocks.0.attn2.processor.to_v_ip.weight']


    names_2 = ["1.to_k_ip.weight", "1.to_v_ip.weight", "3.to_k_ip.weight", "3.to_v_ip.weight", 
               "5.to_k_ip.weight", "5.to_v_ip.weight", "7.to_k_ip.weight", "7.to_v_ip.weight", 
               "9.to_k_ip.weight", "9.to_v_ip.weight", "11.to_k_ip.weight", "11.to_v_ip.weight", 
               "13.to_k_ip.weight", "13.to_v_ip.weight", "15.to_k_ip.weight", "15.to_v_ip.weight", 
               "17.to_k_ip.weight", "17.to_v_ip.weight", "19.to_k_ip.weight", "19.to_v_ip.weight", 
               "21.to_k_ip.weight", "21.to_v_ip.weight", "23.to_k_ip.weight", "23.to_v_ip.weight", 
               "25.to_k_ip.weight", "25.to_v_ip.weight", "27.to_k_ip.weight", "27.to_v_ip.weight", 
               "29.to_k_ip.weight", "29.to_v_ip.weight", "31.to_k_ip.weight", "31.to_v_ip.weight"]
    
    names_2_new = ["1.to_k_ip.0.weight", "1.to_v_ip.0.weight", "3.to_k_ip.0.weight", "3.to_v_ip.0.weight", 
               "5.to_k_ip.0.weight", "5.to_v_ip.0.weight", "7.to_k_ip.0.weight", "7.to_v_ip.0.weight", 
               "9.to_k_ip.0.weight", "9.to_v_ip.0.weight", "11.to_k_ip.0.weight", "11.to_v_ip.0.weight", 
               "13.to_k_ip.0.weight", "13.to_v_ip.0.weight", "15.to_k_ip.0.weight", "15.to_v_ip.0.weight", 
               "17.to_k_ip.0.weight", "17.to_v_ip.0.weight", "19.to_k_ip.0.weight", "19.to_v_ip.0.weight", 
               "21.to_k_ip.0.weight", "21.to_v_ip.0.weight", "23.to_k_ip.0.weight", "23.to_v_ip.0.weight", 
               "25.to_k_ip.0.weight", "25.to_v_ip.0.weight", "27.to_k_ip.0.weight", "27.to_v_ip.0.weight", 
               "29.to_k_ip.0.weight", "29.to_v_ip.0.weight", "31.to_k_ip.0.weight", "31.to_v_ip.0.weight"]

    mapping_1 = {k: v for k, v in zip(names_1, names_2)}
    mapping_2 = {k: v for k, v in zip(names_1, names_2_new)}
    mapping_new = {k: v for k, v in zip(names_2, names_2_new)}

    #ip_root = "results/ip_adapter_plus/vitonhd"
    #ckpt = "checkpoint-6000"
    model_paths = []
    ip_root = args.pretrained_ip_adapter_root_path
    if ip_root.endswith(".bin") and Path(ip_root).exists():
        model_paths.append(ip_root)
    elif ip_root.endswith(".safetensors") and Path(ip_root).exists():
        model_paths.append(ip_root)
    else:
        for folder in os.listdir(ip_root):
            if 'checkpoint' in folder:
                model_paths.append(os.path.join(ip_root, folder, "model.safetensors"))
    
    for model_path in model_paths:
        #model_path = os.path.join(save_dir, "model.bin")
        #model_path = os.path.join(save_dir, "model.safetensors")
        if not(Path(model_path).exists()):
            continue
        if model_path.endswith(".safetensors"):
            sd = load_file(model_path)
        else:
            sd = torch.load(model_path, map_location="cpu")

        image_proj_sd = {}
        ip_sd = {}
        print(f'Starting conversion of {model_path}')
        for k in sd:
            print(k)
            
            #used to convert trained adapter
            # if k.startswith("image_proj_model"):
            #     image_proj_sd[k.replace("image_proj_model.", "")] = sd[k]
            # elif "_ip." in k: 
            #     ip_sd[mapping_2[k.replace("unet.", "")]] = sd[k]
            
            # used to convert already converted weights to new weights to work with diffusers
            # if k.startswith("image_proj"):
            #     image_proj_sd[k.replace("image_proj.", "")] = sd[k]
            # elif "_ip." in k: 
            #     ip_sd[mapping_new[k.replace("ip_adapter.", "")]] = sd[k]  #to translate to new ip-adapter
            
            # used after training with the new weights
            # if k.startswith("image_proj_model"):
            #     image_proj_sd[k.replace("image_proj_model.", "")] = sd[k]
            # elif "_ip." in k: 
            #     ip_sd[mapping_2[k.replace("unet.", "").replace(".0.weight", ".weight")]] = sd[k]
                
            # no idea, probably some legacy code
            # if k.startswith("unet"):
            #     pass
            # elif k.startswith("image_proj_model"):
            #     image_proj_sd[k.replace("image_proj_model.", "")] = sd[k]
            # elif k.startswith("adapter_modules"):
            #     ip_sd[k.replace("adapter_modules.", "")] = sd[k]

        if "ip" in Path(model_path).stem or "adapter" in Path(model_path).stem:
            model_save_path = os.path.join(os.path.dirname(model_path), Path(model_path).stem + "_new.bin")
        else:
            model_save_path = os.path.join(os.path.dirname(model_path), "ip_adapter_new.bin")
        
        #print(image_proj_sd.keys())
        #print(ip_sd.keys())
        #torch.save({"image_proj": image_proj_sd, "ip_adapter": ip_sd}, model_save_path)
        print(f"Saved converted model at {model_save_path}")
        
    print("Done!")