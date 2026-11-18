import os
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
import albumentations as A
import segmentation_models_pytorch as smp
import pytorch_lightning as pl

# --- 1. HELPER: CALCULATE CDR ---
def calculate_cdr(mask_np):
    """Calculates Vertical Cup-to-Disc Ratio"""
    disc_pixels = np.logical_or(mask_np == 1, mask_np == 2)
    cup_pixels = (mask_np == 2)
    
    if np.sum(disc_pixels) == 0 or np.sum(cup_pixels) == 0: 
        return 0.0

    disc_rows, _ = np.where(disc_pixels)
    cup_rows, _ = np.where(cup_pixels)
    
    disc_height = disc_rows.max() - disc_rows.min()
    cup_height = cup_rows.max() - cup_rows.min()
    
    if disc_height == 0: 
        return 0.0
    return cup_height / disc_height

# --- 2. MODEL DEFINITION ---
class OpticDiscCupSegmentationModel(pl.LightningModule):
    def __init__(self, arch="Unet", encoder_name="resnet34", in_channels=3, out_classes=3, **kwargs):
        super().__init__()
        self.model = smp.create_model(
            arch, encoder_name=encoder_name, in_channels=in_channels, classes=out_classes, **kwargs
        )
        params = smp.encoders.get_preprocessing_params(encoder_name)
        self.register_buffer("std", torch.tensor(params["std"]).view(1, 3, 1, 1))
        self.register_buffer("mean", torch.tensor(params["mean"]).view(1, 3, 1, 1))

    def forward(self, image):
        image = (image - self.mean) / self.std
        return self.model(image)

# --- 3. THE COMPARISON FUNCTION ---
def evaluate_comparison(filename, model_baseline, model_final):
    DIR_RAW_IMAGES = "data/Task2_Optic_Disc/01_baseline_original/test"               # Baseline inputs (with vessels)
    DIR_GENAI_IMAGES = "data/Task2_Optic_Disc/04_genai_with_fallback/test"                  # Final pipeline inputs (vessels removed)
    DIR_GT_MASKS = "data/Task2_Optic_Disc/01_baseline_original/testannot"            # Ground truth masks
    
    raw_img_path = os.path.join(DIR_RAW_IMAGES, filename)
    genai_img_path = os.path.join(DIR_GENAI_IMAGES, filename)
    gt_mask_path = os.path.join(DIR_GT_MASKS, filename)

    print(f"\nRunning Comparison for: {filename}")
    print("-" * 50)
    
    # 1. Load Images
    raw_img = cv2.cvtColor(cv2.imread(raw_img_path), cv2.COLOR_BGR2RGB)
    genai_img = cv2.cvtColor(cv2.imread(genai_img_path), cv2.COLOR_BGR2RGB)
    gt_mask_raw = cv2.imread(gt_mask_path, 0)
    
    # 2. Process Ground Truth (0=Bg, 1=Disc, 2=Cup)
    gt_mask = np.zeros_like(gt_mask_raw, dtype=np.uint8)
    gt_mask[gt_mask_raw < 250] = 1 
    gt_mask[gt_mask_raw < 50] = 2  

    transform = A.Compose([A.PadIfNeeded(min_height=384, min_width=480, always_apply=True)])
    
    aug_raw = transform(image=raw_img, mask=gt_mask)
    aug_genai = transform(image=genai_img, mask=gt_mask) # Re-use same mask
    
    img_raw_padded = aug_raw["image"]
    img_genai_padded = aug_genai["image"]
    gt_mask_padded = aug_raw["mask"]
    
    # 4. Prepare Tensors
    def to_tensor(img_np):
        tensor = img_np.astype(np.float32) / 255.0
        tensor = tensor.transpose(2, 0, 1)
        return torch.tensor(tensor, dtype=torch.float32).unsqueeze(0)

    tensor_raw = to_tensor(img_raw_padded)
    tensor_genai = to_tensor(img_genai_padded)
    
    # 5. Run AI Inference
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model_baseline.to(device).eval()
    model_final.to(device).eval()
    tensor_raw = tensor_raw.to(device)
    tensor_genai = tensor_genai.to(device)

    with torch.no_grad():
        # Baseline predicts on Raw Image
        pred_baseline = model_baseline(tensor_raw).softmax(dim=1).argmax(dim=1).squeeze().cpu().numpy()
        # Final pipeline predicts on GenAI Image
        pred_final = model_final(tensor_genai).softmax(dim=1).argmax(dim=1).squeeze().cpu().numpy()

    # 6. Calculate Cup-to-Disc Ratios (CDR)
    gt_cdr = calculate_cdr(gt_mask_padded)
    baseline_cdr = calculate_cdr(pred_baseline)
    final_cdr = calculate_cdr(pred_final)
    
    err_baseline = abs(gt_cdr - baseline_cdr)
    err_final = abs(gt_cdr - final_cdr)

    print(f"Ground Truth CDR: {gt_cdr:.4f}")
    print(f"Baseline CDR:     {baseline_cdr:.4f} (Error: {err_baseline:.4f})")
    print(f"Final GenAI CDR:  {final_cdr:.4f} (Error: {err_final:.4f})")
    print("-" * 50)

    # 7. Plotting the 5-Panel Comparison Graphic
    plt.figure(figsize=(24, 5))
    
    plt.subplot(1, 5, 1)
    plt.imshow(img_raw_padded)
    plt.title("1. Raw Image (Baseline Input)", fontsize=13, fontweight='bold')
    plt.axis("off")

    plt.subplot(1, 5, 2)
    plt.imshow(img_genai_padded) 
    plt.title("2. GenAI Image (Final Input)", fontsize=13, fontweight='bold')
    plt.axis("off")

    plt.subplot(1, 5, 3)
    plt.imshow(gt_mask_padded, cmap='jet', vmin=0, vmax=2, interpolation='nearest')
    plt.title(f"3. Ground Truth\nCDR: {gt_cdr:.3f}", fontsize=13, fontweight='bold')
    plt.axis("off")

    plt.subplot(1, 5, 4)
    plt.imshow(pred_baseline, cmap='jet', vmin=0, vmax=2, interpolation='nearest')
    plt.title(f"4. Baseline Prediction\nCDR: {baseline_cdr:.3f} (Err: {err_baseline:.3f})", fontsize=13, fontweight='bold')
    plt.axis("off")
    
    plt.subplot(1, 5, 5)
    plt.imshow(pred_final, cmap='jet', vmin=0, vmax=2, interpolation='nearest')
    plt.title(f"5. Final Pipeline Prediction\nCDR: {final_cdr:.3f} (Err: {err_final:.3f})", fontsize=13, fontweight='bold')
    plt.axis("off")
    
    plt.tight_layout()
    plt.show()

# --- 4. MAIN EXECUTION BLOCK ---
if __name__ == "__main__":
    
    CKPT_BASELINE = "checkpoints/02_optic_disc_baseline_unet.ckpt"  # Trained on raw images
    CKPT_FINAL = "checkpoints/05_optic_disc_genai_unet.ckpt"        # Trained on GenAI images
    
    print("Loading AI Models...")
    try:
        model_baseline = OpticDiscCupSegmentationModel.load_from_checkpoint(
            CKPT_BASELINE, arch="Unet", encoder_name="resnet34", in_channels=3, out_classes=3)
            
        model_final = OpticDiscCupSegmentationModel.load_from_checkpoint(
            CKPT_FINAL, arch="Unet", encoder_name="resnet34", in_channels=3, out_classes=3)
            
        print("Models loaded successfully!")
        
        test_filename = "v0033.png"
        evaluate_comparison(test_filename, model_baseline, model_final)
        
    except FileNotFoundError as e:
        print(f"Error: Could not find checkpoint. {e}")