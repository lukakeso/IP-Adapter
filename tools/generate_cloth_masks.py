import os
from PIL import Image
import numpy as np
from tqdm import tqdm

root = '/d/hpc/home/lk6760/FRI_HOME/EXPERIMENTS/IP-Adapter'
os.chdir(root)
dataset = "vitonhd"
# dresses = (128, 128, 128)
# lower_body = (0, 128, 128)
# upper_body = (0, 0, 128)
# upper_body cloth for vitonhd = (254, 85, 0)

if dataset == "dresscode":
    for dataset_part, target_label in zip(["dresses", "lower_body", "upper_body"], [(128, 128, 128), (0, 128, 128), (0, 0, 128)]):
        
        #print(dataset_part)
        error_file = os.path.join(root, "datasets", dataset, dataset_part, "gt_cloth_errors.txt")
        images_dir =  os.path.join(root, "datasets", dataset, dataset_part, "images")
        save_dir = os.path.join(root, "datasets", dataset, dataset_part, "gt_cloth_warped_mask")
        os.makedirs(save_dir, exist_ok=True)
        mask_list = []
        save_list = []
        for img_name in os.listdir(images_dir):
            img_path = os.path.join(images_dir, img_name)
            if os.path.isfile(img_path) and img_path.endswith("_0.jpg"):
                mask_list.append(os.path.join(dataset_part, "label_maps", img_name.replace("_0.jpg", "_4.png")))
                save_list.append(os.path.join(save_dir, img_name.replace("_0.jpg", "_0.png")))
        
        assert len(save_list) == len(mask_list)
        
        error_list = []
        for i in tqdm(range(len(mask_list))):
            mask_path = mask_list[i]
            save_path = save_list[i]
            if os.path.isfile(save_path):
                continue
            img = Image.open(os.path.join(root, "datasets", dataset, mask_path)).convert('RGB')
            segmentation_map = np.array(img)
            binary_mask = np.all(segmentation_map == target_label, axis=-1).astype(np.uint8)
            if np.all(binary_mask == 0):
                error_list.append(mask_path)
                continue
                #raise ValueError(f"Pixel value {target_label} was not detected on the image {mask_path}")
            binary_image = Image.fromarray(binary_mask*255)
            binary_image.save(save_path, format='PNG')
            
        print(f"Number of errors: {len(error_list)}")
        with open(error_file, 'w') as file:
            for item in error_list:
                file.write(f"{item}\n")
        #assert len(os.listdir(images_dir)) == 2*len(os.listdir(save_dir))
elif dataset == "vitonhd":
    target_label = (254, 85, 0) #organge upper body cloth
    
    error_file = os.path.join(root, "datasets", dataset, "test", "gt_cloth_errors.txt")
    images_dir =  os.path.join(root, "datasets", dataset, "test", "image-parse-v3")
    save_dir = os.path.join(root, "datasets", dataset, "test", "gt_cloth_warped_mask")
    
    os.makedirs(save_dir, exist_ok=True)
    mask_list = []
    save_list = []
    for img_name in os.listdir(images_dir):
        img_path = os.path.join(images_dir, img_name)
        if os.path.isfile(img_path) and img_path.endswith(".png"):
            mask_list.append(img_path)
            save_list.append(os.path.join(save_dir, img_name))
    
    assert len(save_list) == len(mask_list)
    
    error_list = []
    for i in tqdm(range(len(mask_list))):
        mask_path = mask_list[i]
        save_path = save_list[i]
        if os.path.isfile(save_path):
            continue
        img = Image.open(mask_path).convert('RGB')
        segmentation_map = np.array(img)
        binary_mask = np.all(segmentation_map == target_label, axis=-1).astype(np.uint8)
        if np.all(binary_mask == 0):
            error_list.append(mask_path)
            continue
            #raise ValueError(f"Pixel value {target_label} was not detected on the image {mask_path}")
        binary_image = Image.fromarray(binary_mask*255)
        binary_image.save(save_path, format='PNG')
    
    print(f"Number of errors: {len(error_list)}")
    with open(error_file, 'w') as file:
        for item in error_list:
            file.write(f"{item}\n")