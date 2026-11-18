import os
import cv2
import torch
import numpy as np

from bloodVesselsSegmentation import BloodVesselsSegmentationModel 

CHECKPOINT_PATH = "checkpoints/01_vessel_segmentation.ckpt"
DATA_DIR = "data/Task2_Optic_Disc/01_baseline_original"
MASK_OUTPUT_DIR = "data/Pipeline_Intermediate_Files/generated_vessel_masks(optic disc baseline)"

def generate_masks(model, subset_name):
    input_dir = os.path.join(DATA_DIR, subset_name)
    output_dir = os.path.join(MASK_OUTPUT_DIR, subset_name)
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Processing {subset_name}...")
    
    model.eval()
    device = model.device

    for file_name in os.listdir(input_dir):
        if not file_name.endswith(('.jpg', '.png')): continue
            
        # 1. Load & Preprocess
        img_path = os.path.join(input_dir, file_name)
        original = cv2.imread(img_path)
        if original is None: continue
        
        h_orig, w_orig = original.shape[:2]
        img_rgb = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
        
        # Prepare Tensor
        tensor = torch.from_numpy(img_rgb.astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0)
        
        # Pad to multiple of 32
        h, w = tensor.shape[2:]
        new_h = (h + 31) // 32 * 32
        new_w = (w + 31) // 32 * 32
        tensor = torch.nn.functional.interpolate(tensor, size=(new_h, new_w), mode='bilinear')
        
        # 2. Predict
        with torch.no_grad():
            mask = model(tensor.to(device)).sigmoid().cpu().numpy().squeeze()
            
        # 3. Resize & Save
        mask = cv2.resize(mask, (w_orig, h_orig))
        mask_binary = (mask > 0.5).astype(np.uint8) * 255
        cv2.imwrite(os.path.join(output_dir, file_name), mask_binary)

if __name__ == "__main__":
    model = BloodVesselsSegmentationModel.load_from_checkpoint(
        CHECKPOINT_PATH,
        arch="Unet",
        encoder_name="resnet34",
        in_channels=3,
        out_classes=1
    )
    
    # Move to GPU
    if torch.cuda.is_available(): model.cuda()
    elif torch.backends.mps.is_available(): model.to("mps")
    
    # Run
    for split in ["train", "val", "test"]:
        generate_masks(model, split)