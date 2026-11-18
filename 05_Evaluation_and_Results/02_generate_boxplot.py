import os
import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
import albumentations as A
import segmentation_models_pytorch as smp
import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader

# --- CONFIGURATION ---
DATA_DIR_ORIGINAL = "data/Task2_Optic_Disc/01_baseline_original"
DATA_DIR_INPAINTED = "data/Task2_Optic_Disc/04_genai_with_fallback" 
CKPT_ORIGINAL = "checkpoints/02_optic_disc_baseline_unet.ckpt"
CKPT_INPAINTED = "checkpoints/05_optic_disc_genai_unet.ckpt"

# Use Mac GPU (mps) if available
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"

# --- 2. HELPER FUNCTIONS ---
def calculate_cdr(mask_np):
    """Calculates Vertical Cup-to-Disc Ratio"""
    disc_pixels = np.logical_or(mask_np == 1, mask_np == 2)
    cup_pixels = (mask_np == 2)
    
    if np.sum(disc_pixels) == 0: return 0.0
    if np.sum(cup_pixels) == 0:  return 0.0

    disc_rows, _ = np.where(disc_pixels)
    cup_rows, _ = np.where(cup_pixels)
    
    disc_height = disc_rows.max() - disc_rows.min()
    cup_height = cup_rows.max() - cup_rows.min()
    
    if disc_height == 0: return 0.0
    return cup_height / disc_height

# --- 3. DATASET CLASS ---
class SimpleDataset(Dataset):
    def __init__(self, images_dir, masks_dir):
        self.ids = sorted(os.listdir(images_dir))
        self.images_fps = [os.path.join(images_dir, image_id) for image_id in self.ids]
        self.masks_fps = [os.path.join(masks_dir, image_id) for image_id in self.ids]
        
        # Matches the validation transform from training
        self.transform = A.Compose([A.PadIfNeeded(384, 480)])

    def __getitem__(self, i):
        image = cv2.imread(self.images_fps[i])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(self.masks_fps[i], 0)
        
        # Create masks
        mask_cls = np.zeros_like(mask, dtype=np.uint8)
        mask_cls[mask < 250] = 1 
        mask_cls[mask < 50] = 2  

        # Apply transforms
        sample = self.transform(image=image, mask=mask_cls)
        image, mask_cls = sample["image"], sample["mask"]

        # Normalize
        image = image.astype(np.float32) / 255.0
        
        return (
            torch.tensor(image.transpose(2, 0, 1), dtype=torch.float32), 
            mask_cls, # Keep mask as numpy for easy calculation
            self.ids[i]
        )
    
    def __len__(self):
        return len(self.ids)

# --- 4. MODEL CLASS ---
class OpticDiscCupSegmentationModel(pl.LightningModule):
    def __init__(self, arch, encoder_name, in_channels, out_classes, **kwargs):
        super().__init__()
        self.model = smp.create_model(
            arch, encoder_name=encoder_name, in_channels=in_channels, classes=out_classes, **kwargs
        )
        
        params = smp.encoders.get_preprocessing_params(encoder_name)
        self.register_buffer("std", torch.tensor(params["std"]).view(1, 3, 1, 1))
        self.register_buffer("mean", torch.tensor(params["mean"]).view(1, 3, 1, 1))

    def forward(self, image):
        # Normalize inside the model
        image = (image - self.mean) / self.std
        return self.model(image)

# --- 5. MAIN SCRIPT ---
if __name__ == "__main__":
    print(f"Running on device: {DEVICE}")

    # A. Load Models
    print("Loading models...")
    # We use strict=False as a safety precaution, but the fixed class should allow strict=True
    model_orig = OpticDiscCupSegmentationModel.load_from_checkpoint(CKPT_ORIGINAL, arch="Unet", encoder_name="resnet34", in_channels=3, out_classes=3)
    model_inp = OpticDiscCupSegmentationModel.load_from_checkpoint(CKPT_INPAINTED, arch="Unet", encoder_name="resnet34", in_channels=3, out_classes=3)
    
    model_orig.to(DEVICE); model_orig.eval()
    model_inp.to(DEVICE); model_inp.eval()

    # B. Prepare Data
    ds_orig = SimpleDataset(os.path.join(DATA_DIR_ORIGINAL, "test"), os.path.join(DATA_DIR_ORIGINAL, "testannot"))
    ds_inp = SimpleDataset(os.path.join(DATA_DIR_INPAINTED, "test"), os.path.join(DATA_DIR_INPAINTED, "testannot"))

    errors_original = []
    errors_inpainted = []

    print(f"Calculating CDR errors for {len(ds_orig)} images...")
    
    # Loop through all test images
    for i in range(len(ds_orig)):
        # 1. Get Inputs
        img_orig_tensor, mask_gt, fname = ds_orig[i]
        img_inp_tensor, _, _ = ds_inp[i] # Ground truth is the same

        # 2. Run Inference
        with torch.no_grad():
            # Original Model
            logits_orig = model_orig(img_orig_tensor.unsqueeze(0).to(DEVICE))
            pred_orig = logits_orig.softmax(dim=1).argmax(dim=1).cpu().numpy()[0]
            
            # Inpainted Model
            logits_inp = model_inp(img_inp_tensor.unsqueeze(0).to(DEVICE))
            pred_inp = logits_inp.softmax(dim=1).argmax(dim=1).cpu().numpy()[0]

        # 3. Calculate CDRs
        cdr_gt = calculate_cdr(mask_gt)
        cdr_pred_orig = calculate_cdr(pred_orig)
        cdr_pred_inp = calculate_cdr(pred_inp)

        # 4. Save Absolute Error
        errors_original.append(abs(cdr_pred_orig - cdr_gt))
        errors_inpainted.append(abs(cdr_pred_inp - cdr_gt))

    # C. Plotting
    print("Generating Box Plot...")
    
    # Setup the plot
    plt.figure(figsize=(10, 7))
    data_to_plot = [errors_original, errors_inpainted]
    
    # Create Box Plot
    plt.boxplot(data_to_plot, labels=['Original (With Vessels)', 'Inpainted (No Vessels)'], patch_artist=True)
    
    plt.title('Comparison of CDR Prediction Errors', fontsize=14)
    plt.ylabel('Absolute Error (Lower is Better)', fontsize=12)
    plt.grid(True, axis='y', linestyle='--', alpha=0.7)
    
    # Save file
    save_path = 'thesis_boxplot.png'
    plt.savefig(save_path, dpi=300)
    print(f"Success! Box plot saved as '{save_path}'")
    
    # Optional: Show plot window if supported
    try:
        plt.show()
    except:
        pass