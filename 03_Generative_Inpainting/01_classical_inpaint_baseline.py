import cv2
import os
import numpy as np
from tqdm import tqdm # Optional: for a nice progress bar (pip install tqdm)

# --- CONFIGURATION ---
IMG_DIR_ROOT = "data/Task2_Optic_Disc/01_baseline_original"
MASK_DIR_ROOT = "data/Pipeline_Intermediate_Files/generated_vessel_masks(optic disc baseline)" 
OUTPUT_DIR_ROOT = "data/Pipeline_Intermediate_Files/Classical_Inpaint_output"

SUBSETS = ["train", "val", "test"]

def inpaint_dataset():
    for subset in SUBSETS:
        print(f"\n--- Processing {subset.upper()} set ---")
        
        img_folder = os.path.join(IMG_DIR_ROOT, subset)
        mask_folder = os.path.join(MASK_DIR_ROOT, subset)
        
        out_telea = os.path.join(OUTPUT_DIR_ROOT, subset, "telea")
        out_ns = os.path.join(OUTPUT_DIR_ROOT, subset, "navier_stokes")
        os.makedirs(out_telea, exist_ok=True)
        os.makedirs(out_ns, exist_ok=True)
        
        # Get list of images
        if not os.path.exists(img_folder):
            print(f"Skipping {subset} (Folder not found)")
            continue
            
        file_names = sorted(os.listdir(img_folder))
        
        for file_name in file_names:
            # Construct paths
            img_path = os.path.join(img_folder, file_name)
            mask_path = os.path.join(mask_folder, file_name)
            
            # Check if mask exists
            if not os.path.exists(mask_path):
                continue
                
            # 1. Load Image and Mask
            img = cv2.imread(img_path)
            mask = cv2.imread(mask_path, 0) # Load as grayscale (1 channel)
            
            if img is None or mask is None:
                continue
            
            # 2. Pre-process Mask
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
            mask_dilated = cv2.dilate(mask, kernel, iterations=1)
            
            # 3. Apply Inpainting
            # Method A: Fast Marching Method
            clean_telea = cv2.inpaint(img, mask_dilated, 3, cv2.INPAINT_TELEA)
            
            # Method B: Navier-Stokes
            clean_ns = cv2.inpaint(img, mask_dilated, 3, cv2.INPAINT_NS)
            
            cv2.imwrite(os.path.join(out_telea, file_name), clean_telea)
            cv2.imwrite(os.path.join(out_ns, file_name), clean_ns)
            
        print(f"Completed {subset}. Saved to {OUTPUT_DIR_ROOT}/{subset}")

if __name__ == "__main__":
    inpaint_dataset()
    print("\nAll done!")