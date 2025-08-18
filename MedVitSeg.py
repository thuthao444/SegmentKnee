import torch
import torch.nn as nn
from MedViT import MedViT  # import model gốc

class SegmentationHead(nn.Module):
    def __init__(self, in_channels, num_classes):
        super().__init__()
        self.head = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(in_channels // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // 2, num_classes, kernel_size=1)
        )

    def forward(self, x):
        return self.head(x)

class MedViT2Seg(nn.Module):
    def __init__(self, medvit_model, num_classes=1):
        super().__init__()
        self.backbone = medvit_model
        self.seg_head = SegmentationHead(in_channels=512, num_classes=num_classes)

    def forward(self, x):
        # lấy feature map cuối cùng trước avgpool
        x = self.backbone.stem(x)
        for idx, layer in enumerate(self.backbone.features):
            x = layer(x)
        x = self.backbone.norm(x)  # (B, C, H, W)

        mask = self.seg_head(x)
        return mask
