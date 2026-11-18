import os
import gc
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from skimage.metrics import structural_similarity as ssim

import warnings
warnings.filterwarnings("ignore")

import segmentation_models_pytorch as smp
import pytorch_lightning as pl
from diffusers import StableDiffusionInpaintPipeline

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

# =====================================================================
# 1. MODEL DEFINITIONS
# =====================================================================

class BloodVesselsSegmentationModel(pl.LightningModule):
    """Phase 1: Binary Vessel Segmentation Model"""
    def __init__(self, arch="Unet", encoder_name="resnet34", in_channels=3, out_classes=1, **kwargs):
        super().__init__()
        self.model = smp.create_model(arch, encoder_name=encoder_name, in_channels=in_channels, classes=out_classes, **kwargs)
        params = smp.encoders.get_preprocessing_params(encoder_name)
        self.register_buffer("std", torch.tensor(params["std"]).view(1, 3, 1, 1))
        self.register_buffer("mean", torch.tensor(params["mean"]).view(1, 3, 1, 1))

    def forward(self, image):
        image = (image - self.mean) / self.std
        return self.model(image)

class OpticDiscCupSegmentationModel(pl.LightningModule):
    """Phase 4: Multiclass Optic Disc & Cup Segmentation Model"""
    def __init__(self, arch="Unet", encoder_name="resnet34", in_channels=3, out_classes=3, **kwargs):
        super().__init__()
        self.model = smp.create_model(arch, encoder_name=encoder_name, in_channels=in_channels, classes=out_classes, **kwargs)
        params = smp.encoders.get_preprocessing_params(encoder_name)
        self.register_buffer("std", torch.tensor(params["std"]).view(1, 3, 1, 1))
        self.register_buffer("mean", torch.tensor(params["mean"]).view(1, 3, 1, 1))

    def forward(self, image):
        image = (image - self.mean) / self.std
        return self.model(image)

# =====================================================================
# 2. HELPER FUNCTIONS
# =====================================================================

def calculate_cdr(mask_np):
    """Calculates Vertical Cup-to-Disc Ratio"""
    disc_pixels = np.logical_or(mask_np == 1, mask_np == 2)
    cup_pixels = (mask_np == 2)
    
    if np.sum(disc_pixels) == 0 or np.sum(cup_pixels) == 0: return 0.0

    disc_rows, _ = np.where(disc_pixels)
    cup_rows, _ = np.where(cup_pixels)
    
    disc_height = disc_rows.max() - disc_rows.min()
    cup_height = cup_rows.max() - cup_rows.min()
    
    if disc_height == 0: return 0.0
    return cup_height / disc_height

def clear_mac_memory():
    """Forces the Mac's Unified Memory to clear out massive AI models"""
    gc.collect()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()

# =====================================================================
# 3. THE END-TO-END MASTER PIPELINE
# =====================================================================

def run_clinical_pipeline(image_path, vessel_ckpt, glaucoma_ckpt):
    print("\n" + "="*60)
    print("INITIATING AUTOMATED GLAUCOMA SCREENING PIPELINE")
    print("="*60)

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    ssim_threshold = 0.90

    # ---------------------------------------------------------
    # STEP 0: LOAD RAW PATIENT DATA
    # ---------------------------------------------------------
    print("\n[System] Loading Raw Patient Scan...")
    init_image = Image.open(image_path).convert("RGB").resize((512, 512))
    img_orig_cv = cv2.cvtColor(np.array(init_image), cv2.COLOR_RGB2BGR)
    
    # Prepare tensor for PyTorch (1, 3, 512, 512)
    img_rgb = cv2.cvtColor(img_orig_cv, cv2.COLOR_BGR2RGB)
    tensor_input = torch.from_numpy(img_rgb.astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0).to(device)

    # ---------------------------------------------------------
    # PHASE 1: AUTOMATED VESSEL EXTRACTION
    # ---------------------------------------------------------
    print(f"\n[Phase 1] Loading Vessel U-Net from {vessel_ckpt}...")
    vessel_model = BloodVesselsSegmentationModel.load_from_checkpoint(
        vessel_ckpt, arch="Unet", encoder_name="resnet34", in_channels=3, out_classes=1
    ).to(device)
    vessel_model.eval()

    print("          Extracting Vascular Network...")
    with torch.no_grad():
        vessel_logits = vessel_model(tensor_input)
        vessel_mask_pred = vessel_logits.sigmoid().cpu().numpy().squeeze()
    
    # Convert prediction to binary mask
    vessel_mask_binary = (vessel_mask_pred > 0.5).astype(np.uint8) * 255
    
    # Clean up Phase 1 RAM
    del vessel_model
    clear_mac_memory()

    # ---------------------------------------------------------
    # PHASE 3: GENERATIVE INPAINTING & SSIM ROUTER
    # ---------------------------------------------------------
    print("\n[Phase 3] Initializing Generative Inpainting Engine...")
    pipe = StableDiffusionInpaintPipeline.from_pretrained(
        "runwayml/stable-diffusion-inpainting", torch_dtype=torch.float16, variant="fp16"
    ).to(device)
    pipe.safety_checker = None
    pipe.requires_safety_checker = False
    pipe.enable_attention_slicing()
    pipe.enable_vae_slicing()

    print("          Hallucinating seamless retinal textures...")
    # Dilate mask slightly so AI covers the vessel edges
    mask_dilated_ai = cv2.dilate(vessel_mask_binary, np.ones((7, 7), np.uint8), iterations=1)
    
    prompt = "deep dark orange retinal tissue, seamless medical fundus background, dark red"
    negative_prompt = "optic disc, blood vessels, veins, dark spots, black holes, artificial lenses, cameras, drusen, white dots, lesions, shadows"
    
    output_pil = pipe(
        prompt=prompt, negative_prompt=negative_prompt, image=init_image,
        mask_image=Image.fromarray(mask_dilated_ai), num_inference_steps=30, guidance_scale=4.0      
    ).images[0]

    img_ai_cv = cv2.cvtColor(np.array(output_pil), cv2.COLOR_RGB2BGR)

    print("          Running Alpha-Blend Quality Control (SSIM Router)...")
    img_orig_float = img_orig_cv.astype(np.float32) / 255.0
    img_ai_float = img_ai_cv.astype(np.float32) / 255.0

    mask_refined = cv2.dilate(vessel_mask_binary, np.ones((11, 11), np.uint8), iterations=1)
    mask_blurred = cv2.GaussianBlur((mask_refined.astype(np.float32) / 255.0), (11, 11), 5) 
    mask_3ch = np.stack([mask_blurred]*3, axis=2)

    blended_float = (img_ai_float * mask_3ch) + (img_orig_float * (1.0 - mask_3ch))
    blended_image_cv = (blended_float * 255).astype(np.uint8)

    # SSIM Check against original
    gray_orig = cv2.cvtColor(img_orig_cv, cv2.COLOR_BGR2GRAY)
    gray_blended = cv2.cvtColor(blended_image_cv, cv2.COLOR_BGR2GRAY) 
    score, _ = ssim(gray_orig, gray_blended, full=True)

    if score < ssim_threshold:
        print(f"          Hallucination detected (SSIM: {score:.4f}). Routing to Deterministic Math...")
        mask_full_thick = cv2.dilate(vessel_mask_binary, np.ones((5, 5), np.uint8), iterations=1)
        final_clean_img_cv = cv2.inpaint(img_orig_cv, mask_full_thick, 3, cv2.INPAINT_NS)
        method_used = "Classical Inpaint (Math Fallback)"
    else:
        print(f"          AI passed safety check (SSIM: {score:.4f}).")
        final_clean_img_cv = blended_image_cv
        method_used = "Stable Diffusion (AI)"

    # Clean up Phase 3 RAM
    del pipe
    clear_mac_memory()

    # ---------------------------------------------------------
    # PHASE 4: OPTIC DISC SEGMENTATION & CDR
    # ---------------------------------------------------------
    print(f"\n[Phase 4] Loading Multiclass U-Net from {glaucoma_ckpt}...")
    glaucoma_model = OpticDiscCupSegmentationModel.load_from_checkpoint(
        glaucoma_ckpt, arch="Unet", encoder_name="resnet34", in_channels=3, out_classes=3
    ).to(device)
    glaucoma_model.eval()

    print("          Predicting Optic Disc and Cup Geometry...")
    final_rgb = cv2.cvtColor(final_clean_img_cv, cv2.COLOR_BGR2RGB)
    clean_tensor = torch.from_numpy(final_rgb.astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = glaucoma_model(clean_tensor)
        pred_mask = logits.softmax(dim=1).argmax(dim=1).squeeze().cpu().numpy()

    # Clean up Phase 4 RAM
    del glaucoma_model
    clear_mac_memory()

    # ---------------------------------------------------------
    # STEP 5: CLINICAL OUTPUT & VISUALIZATION
    # ---------------------------------------------------------
    final_cdr = calculate_cdr(pred_mask)
    
    print("\n" + "="*60)
    print(f" FINAL CLINICAL RESULT")
    print(f"   Image Preprocessing : {method_used}")
    print(f"   Calculated CDR      : {final_cdr:.4f}")
    if final_cdr > 0.70:
        print(f"   DIAGNOSTIC FLAG  : High CDR detected. Glaucoma suspect. Human review required.")
    else:
        print(f"   DIAGNOSTIC FLAG  : CDR within normal physiological parameters.")
    print("="*60)

    # Plotting the 4-stage journey
    plt.figure(figsize=(20, 5))
    
    plt.subplot(1, 4, 1)
    plt.imshow(img_rgb)
    plt.title("1. Raw Patient Input", fontsize=14, fontweight='bold')
    plt.axis("off")

    plt.subplot(1, 4, 2)
    plt.imshow(vessel_mask_binary, cmap='gray')
    plt.title("2. Phase 1 Vessel Mask", fontsize=14, fontweight='bold')
    plt.axis("off")

    plt.subplot(1, 4, 3)
    plt.imshow(final_rgb)
    plt.title(f"3. Phase 3 Inpainted\n({method_used})", fontsize=14, fontweight='bold')
    plt.axis("off")

    plt.subplot(1, 4, 4)
    plt.imshow(pred_mask, cmap='jet', vmin=0, vmax=2, interpolation='nearest')
    plt.title(f"4. Phase 4 Geometry\nFinal CDR: {final_cdr:.3f}", fontsize=14, fontweight='bold')
    plt.axis("off")

    plt.tight_layout()
    plt.show()

# =====================================================================
# EXECUTION
# =====================================================================
if __name__ == "__main__":
    
    TEST_IMAGE = "data/Task2_Optic_Disc/01_baseline_original/test/n0026.png"
    
    # The Blood Vessel Model (Phase 1)
    VESSEL_MODEL = "checkpoints/01_vessel_segmentation.ckpt"

    # The GenAI Optic Disc Model (Phase 4)
    GLAUCOMA_MODEL = "checkpoints/05_optic_disc_genai_unet.ckpt"
    
    run_clinical_pipeline(TEST_IMAGE, VESSEL_MODEL, GLAUCOMA_MODEL)