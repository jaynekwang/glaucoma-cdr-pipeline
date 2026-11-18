import os
import shutil 
import cv2
import torch
import numpy as np
import albumentations as A
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torch.utils.data import Dataset as BaseDataset
from torch.optim import lr_scheduler
import segmentation_models_pytorch as smp
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint

# --- CONFIGURATION ---
# --- 1. DATASET CONFIGURATION (CHOOSE YOUR EXPERIMENT) ---
# To test different preprocessing methods, simply uncomment the desired DATA_DIR below.

# Experiment A: Baseline (Raw Images with Blood Vessels)
# DATA_DIR = "data/Task2_Optic_Disc/01_baseline_original"

# Experiment B: Classical Mathematical Inpainting
# NOTE: Navier-Stokes outputs were manually copied to data/Task2_Optic_Disc/02_math_inpainted/ for downstream training
# DATA_DIR = "data/Task2_Optic_Disc/02_math_inpainted"

# Experiment C: Pure Generative AI (No Safety Fallback)
# DATA_DIR = "data/Task2_Optic_Disc/03_genai_no_fallback"

# Experiment D: Final Proposed Pipeline (GenAI with SSIM Fallback) <-- DEFAULT
DATA_DIR = "data/Task2_Optic_Disc/04_genai_with_fallback"
EPOCHS = 100
OUT_CLASSES = 3
BATCH_SIZE = 8

os.makedirs("checkpoints", exist_ok=True)

# --- DIRECTORIES ---
x_train_dir = os.path.join(DATA_DIR, "train")
y_train_dir = os.path.join(DATA_DIR, "trainannot")
x_valid_dir = os.path.join(DATA_DIR, "val")
y_valid_dir = os.path.join(DATA_DIR, "valannot")
x_test_dir = os.path.join(DATA_DIR, "test")
y_test_dir = os.path.join(DATA_DIR, "testannot")

# --- HELPER: Calculate CDR ---
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

# --- DATASET ---
class Dataset(BaseDataset):
    CLASSES = ["background", "disc", "cup"]

    def __init__(self, images_dir, masks_dir, classes=None, augmentation=None):
        self.ids = sorted(os.listdir(images_dir))
        self.images_fps = [os.path.join(images_dir, image_id) for image_id in self.ids]
        self.masks_fps = [os.path.join(masks_dir, image_id) for image_id in self.ids]
        self.augmentation = augmentation

    def __getitem__(self, i):
        image = cv2.imread(self.images_fps[i])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(self.masks_fps[i], 0)
        
        mask_cls = np.zeros_like(mask, dtype=np.uint8)
        mask_cls[mask < 250] = 1 
        mask_cls[mask < 50] = 2  

        if self.augmentation:
            sample = self.augmentation(image=image, mask=mask_cls)
            image, mask_cls = sample["image"], sample["mask"]

        image = image.astype(np.float32) / 255.0
        
        return (
            torch.tensor(image.transpose(2, 0, 1), dtype=torch.float32), 
            torch.tensor(mask_cls, dtype=torch.long), 
            image 
        )
    
    def __len__(self):
        return len(self.ids)

# --- AUGMENTATIONS ---
def get_training_augmentation():
    train_transform = [
        A.HorizontalFlip(p=0.5),
        A.ShiftScaleRotate(scale_limit=0.5, rotate_limit=0, shift_limit=0.1, p=1, border_mode=0),
        A.PadIfNeeded(min_height=320, min_width=320, always_apply=True),
        A.RandomCrop(height=320, width=320, always_apply=True),
        A.GaussNoise(p=0.2),
        A.Perspective(p=0.5),
        A.OneOf([A.CLAHE(p=1), A.RandomBrightnessContrast(p=1), A.RandomGamma(p=1)], p=0.9),
        A.OneOf([A.Sharpen(p=1), A.Blur(blur_limit=3, p=1), A.MotionBlur(blur_limit=3, p=1)], p=0.9),
        A.OneOf([A.RandomBrightnessContrast(p=1), A.HueSaturationValue(p=1)], p=0.9),
    ]
    return A.Compose(train_transform)

def get_validation_augmentation():
    test_transform = [A.PadIfNeeded(384, 480)]
    return A.Compose(test_transform)


# --- LIGHTNING MODULE ---
class OpticDiscCupSegmentationModel(pl.LightningModule):
    def __init__(self, arch, encoder_name, in_channels, out_classes, **kwargs):
        super().__init__()
        self.model = smp.create_model(
            arch, encoder_name=encoder_name, in_channels=in_channels, classes=out_classes, **kwargs
        )
        params = smp.encoders.get_preprocessing_params(encoder_name)
        self.register_buffer("std", torch.tensor(params["std"]).view(1, 3, 1, 1))
        self.register_buffer("mean", torch.tensor(params["mean"]).view(1, 3, 1, 1))
        
        self.loss_fn = smp.losses.DiceLoss(smp.losses.MULTICLASS_MODE, from_logits=True)
        self.training_step_outputs = []
        self.validation_step_outputs = []
        self.test_step_outputs = []

    def forward(self, image):
        image = (image - self.mean) / self.std
        mask = self.model(image)
        return mask

    def shared_step(self, batch, stage):
        image, mask, _ = batch
        logits = self.forward(image)
        loss = self.loss_fn(logits, mask)

        prob_mask = logits.softmax(dim=1)
        pred_mask = prob_mask.argmax(dim=1)
        
        tp, fp, fn, tn = smp.metrics.get_stats(
            pred_mask.long(), mask.long(), mode="multiclass", num_classes=OUT_CLASSES
        )
        return {"loss": loss, "tp": tp, "fp": fp, "fn": fn, "tn": tn, "pred": pred_mask, "gt": mask}

    def shared_epoch_end(self, outputs, stage):
        tp = torch.cat([x["tp"] for x in outputs])
        fp = torch.cat([x["fp"] for x in outputs])
        fn = torch.cat([x["fn"] for x in outputs])
        tn = torch.cat([x["tn"] for x in outputs])

        per_image_iou = smp.metrics.iou_score(tp, fp, fn, tn, reduction="micro-imagewise")
        dataset_iou = smp.metrics.iou_score(tp, fp, fn, tn, reduction="micro")
        f1_score = smp.metrics.f1_score(tp, fp, fn, tn, reduction="micro")
        accuracy = smp.metrics.accuracy(tp, fp, fn, tn, reduction="micro")

        cdr_mae = 0.0
        if stage in ["valid", "test"]:
            errors = []
            for x in outputs:
                preds = x["pred"].cpu().numpy()
                gts = x["gt"].cpu().numpy()
                for i in range(len(preds)):
                    pred_cdr = calculate_cdr(preds[i])
                    gt_cdr = calculate_cdr(gts[i])
                    errors.append(abs(pred_cdr - gt_cdr))
            if errors:
                cdr_mae = np.mean(errors)

        metrics = {
            f"{stage}_per_image_iou": per_image_iou,
            f"{stage}_dataset_iou": dataset_iou,
            f"{stage}_f1_score": f1_score,
            f"{stage}_accuracy": accuracy,
            f"{stage}_cdr_mae": cdr_mae
        }
        self.log_dict(metrics, prog_bar=True)

    def training_step(self, batch, batch_idx):
        info = self.shared_step(batch, "train")
        self.training_step_outputs.append(info)
        return info

    def on_train_epoch_end(self):
        self.shared_epoch_end(self.training_step_outputs, "train")
        self.training_step_outputs.clear()

    def validation_step(self, batch, batch_idx):
        info = self.shared_step(batch, "valid")
        self.validation_step_outputs.append(info)
        return info

    def on_validation_epoch_end(self):
        self.shared_epoch_end(self.validation_step_outputs, "valid")
        self.validation_step_outputs.clear()

    def test_step(self, batch, batch_idx):
        info = self.shared_step(batch, "test")
        self.test_step_outputs.append(info)
        return info

    def on_test_epoch_end(self):
        self.shared_epoch_end(self.test_step_outputs, "test")
        self.test_step_outputs.clear()

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=2e-4)
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS * 100, eta_min=1e-5)
        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "interval": "step", "frequency": 1}}

if __name__ == "__main__":
    
    # 1. Dataset Loading
    train_dataset = Dataset(x_train_dir, y_train_dir, augmentation=get_training_augmentation())
    valid_dataset = Dataset(x_valid_dir, y_valid_dir, augmentation=get_validation_augmentation())
    test_dataset = Dataset(x_test_dir, y_test_dir, augmentation=get_validation_augmentation())

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=True)
    valid_loader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)

    # 2. Model Init
    model = OpticDiscCupSegmentationModel("Unet", "resnet34", in_channels=3, out_classes=OUT_CLASSES)
    
    # 3. Checkpoint & Early Stopping
    checkpoint_callback = ModelCheckpoint(
        monitor='valid_f1_score', 
        dirpath='checkpoints', 
        filename='best_model', 
        save_top_k=1, 
        mode='max',
        save_weights_only=True
    )
    
    early_stop_callback = EarlyStopping(
        monitor="valid_f1_score", 
        patience=10, 
        verbose=True, 
        mode="max" 
    )

    # 4. Trainer Init
    trainer = pl.Trainer(
        max_epochs=EPOCHS, 
        callbacks=[early_stop_callback, checkpoint_callback],
        log_every_n_steps=10,
        accelerator="mps" if torch.backends.mps.is_available() else "auto", 
        precision="16-mixed",
    )

    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=valid_loader)
    
    # 5. Final Results & True Thesis Metrics
    print("\n" + "="*30)
    print(" TRUE THESIS METRICS (Per-Class) ")
    print("="*30)
    
    # Calculate detailed stats on a batch of test data
    images, masks, _ = next(iter(test_loader))
    model.eval()
    with torch.inference_mode():
        logits = model(images.to(model.device))
        pr_masks = logits.softmax(dim=1).argmax(dim=1)
        
        # Get stats for EACH class separately
        tp, fp, fn, tn = smp.metrics.get_stats(
            pr_masks.long(), 
            masks.to(model.device).long(), 
            mode="multiclass", 
            num_classes=3
        )
        
        # Calculate F1 for Background, Disc, and Cup separately
        f1_per_class = smp.metrics.f1_score(tp, fp, fn, tn, reduction="none").mean(dim=0)

    print(f"Optic Disc F1: {f1_per_class[1]:.4f}")
    print(f"Optic Cup F1:  {f1_per_class[2]:.4f}")
    
    # Also print the overall test run for CDR
    test_results = trainer.test(model, dataloaders=test_loader, verbose=False)[0]
    print(f"Average CDR Error: {test_results['test_cdr_mae']:.4f}")
    print("="*30 + "\n")

    # 6. Visualization
    images, masks, image_vis = next(iter(test_loader))
    with torch.inference_mode():
        model.eval()
        logits = model(images.to(model.device))
        pr_masks = logits.softmax(dim=1).argmax(dim=1).cpu()
    
    for idx, (image, gt_mask, image_v, pr_mask) in enumerate(zip(images, masks, image_vis, pr_masks)):
        if idx > 4: break
        plt.figure(figsize=(15, 5))
        
        plt.subplot(1, 3, 1)
        plt.imshow(image_v)
        plt.title("Image")
        plt.axis("off")

        plt.subplot(1, 3, 2)
        plt.imshow(gt_mask.numpy(), cmap='jet', vmin=0, vmax=2, interpolation='nearest')
        plt.title(f"GT (CDR: {calculate_cdr(gt_mask.numpy()):.2f})")
        plt.axis("off")

        plt.subplot(1, 3, 3)
        plt.imshow(pr_mask.numpy(), cmap='jet', vmin=0, vmax=2, interpolation='nearest')
        plt.title(f"Pred (CDR: {calculate_cdr(pr_mask.numpy()):.2f})")
        plt.axis("off")
        plt.show()