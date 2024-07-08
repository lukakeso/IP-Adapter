import os
import json
from PIL import Image
import argparse
from tqdm import tqdm
from transformers import Blip2Processor, Blip2ForConditionalGeneration

def get_caption(model, processor, inputs):
    out = model.generate(**inputs)
    generated_caption = processor.decode(out[0], skip_special_tokens=True).strip()
    generated_caption.replace(" - ", "-")
    return generated_caption

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dataset_name", type=str, default="vitonhd")
    parser.add_argument("-p", "--dataset_part", type=str, default="train", choices=["train", "test"])
    opt = parser.parse_args()
    
    root = '/d/hpc/home/lk6760/FRI_HOME/EXPERIMENTS/IP-Adapter'
    os.chdir(root)
    processor = Blip2Processor.from_pretrained("models/Salesforce/blip2-opt-2.7b")
    model = Blip2ForConditionalGeneration.from_pretrained("models/Salesforce/blip2-opt-2.7b", device_map=0)

    img_list = []
    ref_list = []
    mask_list = []
    if opt.dataset_name == "vitonhd":
        images_dir =  os.path.join(root, f"datasets/vitonhd/{opt.dataset_part}")
        ref_dir = os.path.join(root, f"datasets/vitonhd/{opt.dataset_part}")
        dataset_file = os.path.join(root, f"datasets/vitonhd/{opt.dataset_part}_pairs.txt")

        with open(dataset_file, 'r') as f:
            for line in f.readlines():
                img_name, _ = line.strip().split()
                img_list.append(os.path.join("image", img_name))
                ref_list.append(os.path.join("cloth", img_name))
                #mask_list.append(os.path.join("gt_cloth_warped_mask", img_name))
                mask_list.append(os.path.join("agnostic-mask", img_name.split(".jpg")[0]+"_mask.png")) #for testing/inference

    elif opt.dataset_name == "dresscode":
        images_dir =  os.path.join(root, "datasets/dresscode")
        ref_dir = os.path.join(root, "datasets/dresscode")
        if opt.dataset_part == "test":
            dataset_file = os.path.join(root, f"datasets/dresscode/test_pairs_paired.txt")
        else:
            dataset_file = os.path.join(root, f"datasets/dresscode/train_pairs_filtered.txt")
        with open(dataset_file, 'r') as f:
            for line in f.readlines():
                img_name, ref_name, category = line.strip().split()
                if category == "0":
                    dataset_part = "upper_body"
                elif category == "1":
                    dataset_part = "lower_body"
                elif category == "2":
                    dataset_part = "dresses"
                img_list.append(os.path.join(dataset_part, "images", img_name))
                ref_list.append(os.path.join(dataset_part, "images", ref_name))
                #mask_list.append(os.path.join(dataset_part, "gt_cloth_warped_mask", img_name.replace("jpg", "png")))
                mask_list.append(os.path.join(dataset_part, "mask", img_name.replace("jpg", "png")))


    assert len(img_list) == len(ref_list)
    assert len(img_list) == len(mask_list)
    
    dataset_pairs = []
    for i in tqdm(range(len(img_list))):
        all_imgs=[]
        img_name = img_list[i]
        mask_name = mask_list[i]
        ref_name = ref_list[i]
        
        #img = Image.open(os.path.join(root, images_dir, img_name)).convert('RGB')
        ref = Image.open(os.path.join(root, ref_dir, ref_name)).convert('RGB')
        
        # print(img_name, ref_name)
        # img_inputs = processor(img, return_tensors="pt", max_length=15, truncation=True).to("cuda")
        # print("IMG:", get_caption(model, processor, img_inputs))
        ref_inputs = processor(ref, return_tensors="pt", max_length=15, truncation=True).to("cuda")
        caption = get_caption(model, processor, ref_inputs)
        
        # # [{"image_file": "1.png", "text": "A dog"}]
        dataset_pairs.append({"cloth_file": ref_name, 
                              "mask_file": mask_name, 
                              "image_file": img_name, 
                              "text": caption})
        
    save_file = open(f'datasets/{opt.dataset_name}/{opt.dataset_part}_data.json', "w")  
    json.dump(dataset_pairs, save_file)  
    save_file.close()