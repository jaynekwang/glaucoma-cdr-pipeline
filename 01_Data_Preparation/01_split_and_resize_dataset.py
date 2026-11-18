import os
import shutil
import random
from PIL import Image

# --- CONFIGURATION ---
# This script is used for both datasets. Set the three paths below to match which
# dataset you're preparing, then run it. The script performs a 70 / 15 / 15
# train/val/test split and resizes all images and masks to 512×512.
#
#   Task 1 — Amalgamated vessel dataset (DRIVE + CHASE_DB1 + FIVES):
#       INPUT_IMAGES = "data/Task1_Blood_Vessels/00_raw_input/images"
#       INPUT_MASKS  = "data/Task1_Blood_Vessels/00_raw_input/masks"
#       OUTPUT_DIR   = "data/Task1_Blood_Vessels/01_training_data"
#
#   Task 2 — REFUGE2 optic disc / cup dataset:
#       INPUT_IMAGES = "data/Task2_Optic_Disc/00_raw_input/images"
#       INPUT_MASKS  = "data/Task2_Optic_Disc/00_raw_input/masks"
#       OUTPUT_DIR   = "data/Task2_Optic_Disc/01_baseline_original"

INPUT_IMAGES = "data/Task2_Optic_Disc/00_raw_input/images"
INPUT_MASKS  = "data/Task2_Optic_Disc/00_raw_input/masks"
OUTPUT_DIR   = "data/Task2_Optic_Disc/01_baseline_original"

def split_dataset():
    # Create the required PyTorch directories
    folders = ['train', 'trainannot', 'val', 'valannot', 'test', 'testannot']
    for folder in folders:
        os.makedirs(os.path.join(OUTPUT_DIR, folder), exist_ok=True)

    # Get all images
    all_images = [f for f in os.listdir(INPUT_IMAGES) if f.endswith(('.png', '.jpg'))]
    
    random.seed(42)
    random.shuffle(all_images)

    # Calculate split indices (70% / 15% / 15%)
    total = len(all_images)
    train_split = int(total * 0.70)
    val_split = train_split + int(total * 0.15)

    train_files = all_images[:train_split]
    val_files = all_images[train_split:val_split]
    test_files = all_images[val_split:]

    def copy_files(file_list, img_dest, mask_dest):
        for filename in file_list:
            mask_name = filename
            
            src_img = os.path.join(INPUT_IMAGES, filename)
            src_mask = os.path.join(INPUT_MASKS, mask_name)

            if os.path.exists(src_mask):
                shutil.copy(src_img, os.path.join(OUTPUT_DIR, img_dest, filename))
                shutil.copy(src_mask, os.path.join(OUTPUT_DIR, mask_dest, mask_name))
            else:
                print(f"Could not find mask for {filename}")

    print("Copying Training files...")
    copy_files(train_files, 'train', 'trainannot')
    print("Copying Validation files...")
    copy_files(val_files, 'val', 'valannot')
    print("Copying Testing files...")
    copy_files(test_files, 'test', 'testannot')

    print(f"Dataset split into {OUTPUT_DIR}")
    print(f"Train: {len(train_files)} | Val: {len(val_files)} | Test: {len(test_files)}")

    print("Resizing all images to 512x512...")
    img_folders  = ['train', 'val', 'test']
    mask_folders = ['trainannot', 'valannot', 'testannot']

    for folder in img_folders:
        for filename in os.listdir(os.path.join(OUTPUT_DIR, folder)):
            if filename.endswith(('.png', '.jpg')):
                path = os.path.join(OUTPUT_DIR, folder, filename)
                Image.open(path).convert("RGB").resize((512, 512), Image.LANCZOS).save(path)

    for folder in mask_folders:
        for filename in os.listdir(os.path.join(OUTPUT_DIR, folder)):
            if filename.endswith(('.png', '.jpg')):
                path = os.path.join(OUTPUT_DIR, folder, filename)
                Image.open(path).resize((512, 512), Image.NEAREST).save(path)

    print("Done.")

if __name__ == "__main__":
    split_dataset()