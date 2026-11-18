import os
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
from pytorch_lightning.callbacks import EarlyStopping

from bloodVesselsSegmentation import BloodVesselsSegmentationModel

# -------------------------------------------------------------------------
# GLOBAL CONSTANTS & PATHS
# -------------------------------------------------------------------------
DATA_DIR = "data/Task1_Blood_Vessels/01_training_data"

x_train_dir = os.path.join(DATA_DIR, "train")
y_train_dir = os.path.join(DATA_DIR, "trainannot")

x_valid_dir = os.path.join(DATA_DIR, "val")
y_valid_dir = os.path.join(DATA_DIR, "valannot")

x_test_dir = os.path.join(DATA_DIR, "test")
y_test_dir = os.path.join(DATA_DIR, "testannot")

# -------------------------------------------------------------------------
# DATASET CLASS
# -------------------------------------------------------------------------
class Dataset(BaseDataset):

    CLASSES = ["blood vessels", "background"]

    def __init__(
        self,
        images_dir,
        masks_dir,
        classes="blood vessels",
        augmentation=None,
    ):
        self.ids = sorted(os.listdir(images_dir))
        self.images_fps = [os.path.join(images_dir, image_id) for image_id in self.ids]
        self.masks_fps = [os.path.join(masks_dir, image_id) for image_id in self.ids]

        self.augmentation = augmentation

    def __getitem__(self, i):
        image = cv2.imread(self.images_fps[i])
        mask = cv2.imread(self.masks_fps[i], 0)
        # BGR-->RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_vis = image.copy()
        image = image.astype(np.float32) / 255.0
        

        # extract certain classes from mask
        mask = (mask > 127).astype("float32")  # binary mask
        mask = np.expand_dims(mask, axis=-1)

        # apply augmentations
        if self.augmentation:
            # convert back to uint8 for albumentations
            image_uint8 = (image * 255).astype(np.uint8)
            sample = self.augmentation(image=image_uint8, mask=mask)
            image, mask = sample["image"], sample["mask"]
            # normalize to float32 after augmentation
            image = image.astype(np.float32) / 255.0
        
        return torch.tensor(image.transpose(2,0,1), dtype=torch.float32), \
                torch.tensor(mask.transpose(2,0,1), dtype=torch.float32), image_vis

    def __len__(self):
        return len(self.ids)
    
# -------------------------------------------------------------------------
# HELPER FUNCTIONS
# -------------------------------------------------------------------------
def visualize(**images):
    """Plot images in one row."""
    n = len(images)
    plt.figure(figsize=(16, 5))
    for i, (name, image) in enumerate(images.items()):
        plt.subplot(1, n, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.title(" ".join(name.split("_")).title())
        if name == "image":
            plt.imshow(image.transpose(1, 2, 0))
        else:
            plt.imshow(image)
    plt.show()

def get_training_augmentation():
    train_transform = [
        A.HorizontalFlip(p=0.5),
        A.ShiftScaleRotate(
            scale_limit=0.5, rotate_limit=0, shift_limit=0.1, p=1, border_mode=0
        ),
        A.PadIfNeeded(min_height=320, min_width=320, always_apply=True),
        A.RandomCrop(height=320, width=320, always_apply=True),
        A.GaussNoise(p=0.2),
        A.Perspective(p=0.5),
        A.OneOf(
            [
                A.CLAHE(p=1),
                A.RandomBrightnessContrast(p=1),
                A.RandomGamma(p=1),
            ],
            p=0.9,
        ),
        A.OneOf(
            [
                A.Sharpen(p=1),
                A.Blur(blur_limit=3, p=1),
                A.MotionBlur(blur_limit=3, p=1),
            ],
            p=0.9,
        ),
        A.OneOf(
            [
                A.RandomBrightnessContrast(p=1),
                A.HueSaturationValue(p=1),
            ],
            p=0.9,
        ),
    ]
    return A.Compose(train_transform)

def get_validation_augmentation():
    """Add paddings to make image shape divisible by 32"""
    test_transform = [
        A.PadIfNeeded(384, 480),
    ]
    return A.Compose(test_transform)

# -------------------------------------------------------------------------
# MAIN EXECUTION
# -------------------------------------------------------------------------
if __name__ == "__main__":
    CLASSES = ["blood vessels"]

    train_dataset = Dataset(
        x_train_dir,
        y_train_dir,
        augmentation=get_training_augmentation(),
        classes=CLASSES,
    )

    valid_dataset = Dataset(
        x_valid_dir,
        y_valid_dir,
        augmentation=get_validation_augmentation(),
        classes=CLASSES,
    )

    test_dataset = Dataset(
        x_test_dir,
        y_test_dir,
        augmentation=get_validation_augmentation(),
        classes=CLASSES,
    )

    train_loader = DataLoader(
        train_dataset, 
        batch_size=16, 
        shuffle=True, 
        num_workers=4,
        pin_memory=True,      
        persistent_workers=True 
    )
    valid_loader = DataLoader(
        valid_dataset, 
        batch_size=16, 
        shuffle=False, 
        num_workers=4,
        pin_memory=True,     
        persistent_workers=True 
    )
    test_loader = DataLoader(
        test_dataset, 
        batch_size=16, 
        shuffle=False, 
        num_workers=4,
        pin_memory=True,      
        persistent_workers=True 
    )

    # Some training hyperparameters
    EPOCHS = 100 
    T_MAX = EPOCHS * len(train_loader)
    OUT_CLASSES = 1

    # Instantiate Model
    model = BloodVesselsSegmentationModel("Unet", "resnet34", in_channels=3, out_classes=OUT_CLASSES)
    
    # Pass T_MAX to the model instance so the scheduler works correctly
    model.t_max = T_MAX

    early_stop_callback = EarlyStopping(
        monitor="val_loss", 
        min_delta=0.00, 
        patience=10, 
        verbose=True, 
        mode="min"
    )

    trainer = pl.Trainer(
        max_epochs=EPOCHS, 
        callbacks=[early_stop_callback],
        log_every_n_steps=1,
        accelerator="auto", 
        precision = "16-mixed",
    )

    trainer.fit(
        model,
        train_dataloaders=train_loader,
        val_dataloaders=valid_loader,
    )

    # run validation dataset
    valid_metrics = trainer.validate(model, dataloaders=valid_loader, verbose=False)
    print(valid_metrics)

    # run test dataset
    test_metrics = trainer.test(model, dataloaders=test_loader, verbose=False)
    print(test_metrics)

    # --- VISUALIZATION BLOCK ---
    images, masks, image_vis = next(iter(test_loader))
    with torch.inference_mode():
        model.eval()
        logits = model(images)
    pr_masks = logits.sigmoid()
    
    for idx, (image, gt_mask, image_vis, pr_mask) in enumerate(zip(images, masks, image_vis, pr_masks)):
        if idx <= 4:
            gt_np = gt_mask.numpy().squeeze()
            pr_np = pr_mask.numpy().squeeze()
            
            pr_bin = (pr_np > 0.5).astype(float)
            gt_bin = (gt_np > 0.5).astype(float)
            
            tp = np.logical_and(pr_bin == 1, gt_bin == 1)
            fp = np.logical_and(pr_bin == 1, gt_bin == 0)
            fn = np.logical_and(pr_bin == 0, gt_bin == 1)
            
            diff_map = np.zeros((gt_bin.shape[0], gt_bin.shape[1], 3))
            diff_map[tp] = [1, 1, 1] 
            diff_map[fp] = [0, 1, 0] 
            diff_map[fn] = [1, 0, 0] 

            plt.figure(figsize=(20, 5))
            plt.subplot(1, 4, 1); plt.imshow(image_vis); plt.title("Image"); plt.axis("off")
            plt.subplot(1, 4, 2); plt.imshow(gt_np, cmap="gray"); plt.title("Ground truth"); plt.axis("off")
            plt.subplot(1, 4, 3); plt.imshow(pr_np, cmap="gray"); plt.title("Prediction"); plt.axis("off")
            plt.subplot(1, 4, 4); plt.imshow(diff_map); plt.title("Diff Map\n(Red=Missed, Green=Noise)"); plt.axis("off")
            plt.show()
        else:
            break