import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
from tqdm import tqdm
import argparse
from PIL import Image
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
from MedViT import MedViT_tiny, MedViT_small, MedViT_base, MedViT_large
from torchvision import transforms
from sklearn.model_selection import train_test_split
from MedVitSeg import MedViT2Seg 

# -----------------------
# Model mapping
# -----------------------
model_classes = {
    'MedViT_tiny': MedViT_tiny,
    'MedViT_small': MedViT_small,
    'MedViT_base': MedViT_base,
    'MedViT_large': MedViT_large
}

# -----------------------
# Custom Dataset
# -----------------------
class CustomSegDataset(data.Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.images = sorted([f for f in os.listdir(image_dir) if f.lower().endswith(('.png','.jpg','.jpeg'))])
        self.masks = sorted([f for f in os.listdir(mask_dir) if f.lower().endswith(('.png','.jpg','.jpeg'))])
        assert len(self.images) == len(self.masks), "Number of images and masks do not match"

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # load ảnh và mask
        img_path = os.path.join(self.image_dir, self.images[idx])
        mask_path = os.path.join(self.mask_dir, self.masks[idx])

        image = np.array(Image.open(img_path).convert('RGB'))
        mask = np.array(Image.open(mask_path).convert('L'))

        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']

        # convert mask sang long tensor và clamp giá trị
        if not isinstance(mask, torch.Tensor):
            mask = torch.from_numpy(mask).long()
        else:
            mask = mask.long()
            
        image = image.float() / 255.0

        return image, mask



criterion = nn.CrossEntropyLoss(ignore_index=0)

# -----------------------
# IoU metric
# -----------------------
def iou_score(pred, target, num_classes=5):
    pred_classes = torch.argmax(pred, dim=1)
    ious = []
    for cls in range(1, num_classes):  # start từ 1, bỏ background
        pred_cls = (pred_classes == cls).float()
        target_cls = (target == cls).float()
        intersection = (pred_cls * target_cls).sum()
        union = pred_cls.sum() + target_cls.sum() - intersection
        if union == 0:
            ious.append(torch.tensor(1.0, device=pred.device))
        else:
            ious.append(intersection / union)
    return torch.mean(torch.stack(ious)).item()

# -----------------------
# Training function
# -----------------------
def train_segmentation(epochs, net, train_loader, val_loader, optimizer, scheduler, device, save_path, num_classes):
    best_iou = 0.0

    for epoch in range(epochs):
        net.train()
        running_loss = 0.0
        train_bar = tqdm(train_loader, file=sys.stdout)
        for images, masks in train_bar:
            images, masks = images.to(device), masks.to(device)
            optimizer.zero_grad()
            outputs = net(images)  # [B, num_classes, H, W]
            loss = criterion(outputs, masks)  # CE loss
            loss.backward()
            optimizer.step()
            scheduler.step()
            running_loss += loss.item()
            train_bar.desc = f"Epoch[{epoch+1}/{epochs}] loss:{loss:.4f}"

        # Validation
        net.eval()
        val_loss = 0.0
        val_iou = 0.0
        with torch.no_grad():
            val_bar = tqdm(val_loader, file=sys.stdout)
            for images, masks in val_bar:
                images, masks = images.to(device), masks.to(device)
                outputs = net(images)
                val_loss += criterion(outputs, masks).item()
                val_iou += iou_score(outputs, masks, num_classes)

        val_loss /= len(val_loader)
        val_iou /= len(val_loader)

        print(f"Epoch [{epoch+1}/{epochs}] Train Loss: {running_loss/len(train_loader):.4f} "
              f"Val Loss: {val_loss:.4f} Val IoU: {val_iou:.4f}")

        # Save best model
        if val_iou > best_iou:
            print("Saving best checkpoint...")
            best_iou = val_iou
            state = {
                'model': net.state_dict(),
                'optimizer': optimizer.state_dict(),
                'lr_scheduler': scheduler.state_dict(),
                'best_iou': best_iou,
                'epoch': epoch,
            }
            torch.save(state, save_path)
    print("Finished Training")


# -----------------------
# Main
# -----------------------
def main(args):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_transform = A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.15),   # xoay ngẫu nhiên ~15%
        A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.0, p=0.5),
        A.ToGray(p=0.15),  # chuyển grayscale với xác suất 15%
        A.Resize(256, 256),  # resize về cùng kích thước
        ToTensorV2(),
    ])

    val_transform = A.Compose([
        A.Resize(256, 256),
        ToTensorV2(),
    ])

    img_dir = os.path.join(args.dataset, "train")
    mask_dir = os.path.join(args.dataset, "masks")
    images = sorted([f for f in os.listdir(img_dir) if f.lower().endswith(('.png','.jpg','.jpeg'))])
    masks = sorted([f for f in os.listdir(mask_dir) if f.lower().endswith(('.png','.jpg','.jpeg'))])

    train_imgs, val_imgs, train_masks, val_masks = train_test_split(
        images, masks, test_size=0.2, random_state=42
    )

    # dataset
    train_dataset = CustomSegDataset(
        image_dir=img_dir,
        mask_dir=mask_dir,
        transform=train_transform
    )
    train_dataset.images = train_imgs
    train_dataset.masks = train_masks

    val_dataset = CustomSegDataset(
        image_dir=img_dir,
        mask_dir=mask_dir,
        transform=val_transform
    )
    val_dataset.images = val_imgs
    val_dataset.masks = val_masks

    train_loader = data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    num_classes = 5  # binary segmentation

    model_class = model_classes.get(args.model_name)
    backbone = model_class()  

    net = MedViT2Seg(backbone, num_classes=num_classes).to(device)

    optimizer = optim.AdamW(net.parameters(), lr=args.lr, weight_decay=0.05)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs*len(train_loader), eta_min=1e-6)

    save_path = f'./{args.model_name}_seg.pth'
    train_segmentation(args.epochs, net, train_loader, val_loader, optimizer, scheduler, device, save_path, num_classes)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train MedViT for Segmentation")
    parser.add_argument('--model_name', type=str, default='MedViT_tiny', help='Model name')
    parser.add_argument('--dataset', type=str, required=True, help='Path to dataset folder')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    args = parser.parse_args()
    main(args)
