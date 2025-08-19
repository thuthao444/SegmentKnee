import torch
import torch.nn as nn
from MedViT import MedViT  # import model gốc
import torch.nn.functional as F  # --- THÊM DÒNG NÀY ---

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

        # --- SỬA 1: tự suy ra số kênh đầu ra cuối backbone ---
        in_channels = getattr(self.backbone.norm, "num_features", None)
        if in_channels is None:
            raise ValueError("Không suy ra được số kênh output từ backbone. Kiểm tra self.backbone.norm.num_features.")
        self.seg_head = SegmentationHead(in_channels=in_channels, num_classes=num_classes)

    def forward(self, x):
        # --- SỬA 2: ghi lại size gốc để upsample logits ---
        H, W = x.shape[-2:]

        # lấy feature map cuối cùng trước avgpool
        x = self.backbone.stem(x)
        for idx, layer in enumerate(self.backbone.features):
            x = layer(x)
        x = self.backbone.norm(x)  # (B, C, h, w)

        mask = self.seg_head(x)    # (B, num_classes, h, w)

        # Upsample về đúng (H, W) của ảnh/mask đầu vào để CE loss không bị mismatch
        mask = F.interpolate(mask, size=(H, W), mode="bilinear", align_corners=False)
        return mask
