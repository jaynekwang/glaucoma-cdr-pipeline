import os
import torch
import cv2
import numpy as np
from PIL import Image
from diffusers import StableDiffusionInpaintPipeline
from skimage.metrics import structural_similarity as ssim

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

# --- CONFIGURATION ---
BASE_INPUT_DIR = "data/Task2_Optic_Disc/01_baseline_original" 
BASE_MASK_DIR = "data/Pipeline_Intermediate_Files/generated_vessel_masks(optic disc baseline)" 
BASE_OUTPUT_DIR = "data/Task2_Optic_Disc/04_genai_with_fallback" 

SPLITS = ["train", "test", "val"]

DEVICE = "mps" 
SSIM_THRESHOLD = 0.90 # The safety cutoff limit

def process_single_folder(input_dir, mask_dir, output_dir, pipe):
    os.makedirs(output_dir, exist_ok=True)
    image_files = [f for f in os.listdir(input_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    total_images = len(image_files)
    
    for index, filename in enumerate(image_files, start=1):
        print(f"Processing [{index}/{total_images}]: {filename}")
        
        image_path = os.path.join(input_dir, filename)
        mask_path = os.path.join(mask_dir, filename)
        output_path = os.path.join(output_dir, filename)

        if not os.path.exists(mask_path):
            print(f"WARNING: No matching mask found for {filename}. Skipping...")
            continue

        try:
            # --- 1. LOAD IMAGES ---
            init_image = Image.open(image_path).convert("RGB").resize((512, 512))
            img_orig_cv = cv2.cvtColor(np.array(init_image), cv2.COLOR_RGB2BGR)
            img_orig_float = img_orig_cv.astype(np.float32) / 255.0
            
            mask_cv = cv2.imread(mask_path, 0)
            mask_cv = cv2.resize(mask_cv, (512, 512))

            max_val = mask_cv.max()
            if max_val <= 1: mask_cv = mask_cv * 255
            elif max_val < 128: mask_cv = cv2.normalize(mask_cv, None, 0, 255, cv2.NORM_MINMAX)
            _, mask_thresh = cv2.threshold(mask_cv, 127, 255, cv2.THRESH_BINARY)

            # --- 2. PREPARE MASK FOR AI ---
            kernel_ai = np.ones((7, 7), np.uint8) 
            mask_dilated_ai = cv2.dilate(mask_thresh, kernel_ai, iterations=1)
            mask_image_pil = Image.fromarray(mask_dilated_ai)

            # --- 3. RUN AI SCENE COMPLETION ---
            prompt = "deep dark orange retinal tissue, seamless medical fundus background, dark red"
            negative_prompt = "optic disc, blood vessels, veins, dark spots, black holes, artificial lenses, cameras, drusen, white dots, lesions, shadows"
            output_pil = pipe(
                prompt=prompt,
                negative_prompt=negative_prompt,
                image=init_image,
                mask_image=mask_image_pil,
                num_inference_steps=30, 
                guidance_scale=4.0      
            ).images[0]

            img_ai_cv = cv2.cvtColor(np.array(output_pil), cv2.COLOR_RGB2BGR)
            img_ai_float = img_ai_cv.astype(np.float32) / 255.0

            # --- 4. PREPARE SEAMLESS ALPHA BLEND ---
            kernel_blend = np.ones((11, 11), np.uint8) 
            mask_refined = cv2.dilate(mask_thresh, kernel_blend, iterations=1)
            mask_float = mask_refined.astype(np.float32) / 255.0
            mask_blurred = cv2.GaussianBlur(mask_float, (11, 11), 5) 
            mask_3ch = np.stack([mask_blurred]*3, axis=2)

            blended_float = (img_ai_float * mask_3ch) + (img_orig_float * (1.0 - mask_3ch))
            blended_image_cv = (blended_float * 255).astype(np.uint8)

            # --- 5. GLOBAL SSIM QUALITY CONTROL CHECK ---
            gray_orig = cv2.cvtColor(img_orig_cv, cv2.COLOR_BGR2GRAY)
            gray_blended = cv2.cvtColor(blended_image_cv, cv2.COLOR_BGR2GRAY) 
            score, _ = ssim(gray_orig, gray_blended, full=True)
            
            # --- 6. THE FALLBACK ROUTER ---
            if score < SSIM_THRESHOLD:
                print(f"SSIM: {score:.4f} | Hallucination detected. Routing to Navier-Stokes...")
                mask_full_thick = cv2.dilate(mask_thresh, np.ones((5, 5), np.uint8), iterations=1)
                final_image = cv2.inpaint(img_orig_cv, mask_full_thick, 3, cv2.INPAINT_NS)
            else:
                print(f"SSIM: {score:.4f} | AI passed safety check.")
                final_image = blended_image_cv

            cv2.imwrite(output_path, final_image)

        except Exception as e:
            print(f"FAILED on {filename}: {e}")

def run_full_dataset():
    # --- 0. SETUP AND LOAD AI ONLY ONCE ---
    print(f"Loading AI Model on {DEVICE}.")
    
    try:
        pipe = StableDiffusionInpaintPipeline.from_pretrained(
            "runwayml/stable-diffusion-inpainting",
            torch_dtype=torch.float16,
            variant="fp16"
        )
        pipe = pipe.to(DEVICE)
        pipe.safety_checker = None
        pipe.requires_safety_checker = False
        pipe.enable_attention_slicing() 
        pipe.enable_vae_slicing()
        print("Model loaded successfully!\n")
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # --- LOOP THROUGH TRAIN, TEST, AND VAL ---
    for split in SPLITS:
        print("=" * 50)
        print(f"STARTING SPLIT: {split.upper()}")
        print("=" * 50)
        
        current_input_dir = os.path.join(BASE_INPUT_DIR, split)
        current_mask_dir = os.path.join(BASE_MASK_DIR, split)
        current_output_dir = os.path.join(BASE_OUTPUT_DIR, split)

        if not os.path.exists(current_input_dir):
            print(f"Directory not found: {current_input_dir}. Skipping...")
            continue
            
        process_single_folder(current_input_dir, current_mask_dir, current_output_dir, pipe)

    print("\n" + "=" * 50)
    print(f"All Done! All splits saved in: {BASE_OUTPUT_DIR}")

if __name__ == "__main__":
    run_full_dataset()